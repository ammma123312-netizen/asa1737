"""Stage a node's share of a safetensors checkpoint as sparse local files.

Why: reading pre-staged experts back through FUSE costs a Python round trip per
kernel request; on runners that capped decode at ~78 s per forward. A worker in
Colibri's cluster only ever reads (a) each shard's header and (b) the routed
experts it owns, so it can run from local files that contain exactly those bytes
at their original offsets and holes everywhere else. The engine then reads at
disk / page-cache speed, with no FUSE in the hot path.

    # worker k of n: headers + owned experts ((expert + layer) % n == k)
    python -m asa.swarm.stage deepseek-ai/DeepSeek-V4.1-Flash /srv/view --owner 3/16
    # coordinator: everything except routed experts; engram shards linked to an HFFS mount
    python -m asa.swarm.stage deepseek-ai/DeepSeek-V4.1-Flash /srv/view \\
        --include . --exclude '^layers\\.\\d+\\.ffn\\.experts\\.|engram' \\
        --link-from /srv/hffs-mount --link '*-0004[78]-of-*'

Holes read as zeros, so a staged view is only valid for a node that never reads
outside its selection; `verify_selection` in the tests checks that on a fixture.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from asa.swarm.hffs import HubRepo, owner_selector, parse_owner, regex_selector

CHUNK = 16 << 20       # split big tensors so a single file downloads in parallel
COALESCE_GAP = 1 << 20  # merge neighbouring tensors closer than this into one request


class LocalRepo:
    """Same interface as HubRepo, reading from a local directory (tests, mirrors)."""

    def __init__(self, root: str):
        self.root = root
        self.files = {}
        for dp, _, fs in os.walk(root):
            for f in fs:
                p = os.path.join(dp, f)
                self.files[os.path.relpath(p, root)] = os.path.getsize(p)

    def read_range(self, path: str, start: int, end: int) -> bytes:
        with open(os.path.join(self.root, path), "rb") as f:
            f.seek(start)
            return f.read(end - start)


def header_of(repo, path: str) -> tuple[bytes, dict]:
    n = struct.unpack("<Q", repo.read_range(path, 0, 8))[0]
    raw = repo.read_range(path, 0, 8 + n)
    return raw, json.loads(raw[8:])


def plan_ranges(header: dict, base: int, select) -> list[tuple[int, int]]:
    """Coalesced, chunked absolute byte ranges of the selected tensors."""
    spans = sorted((base + m["data_offsets"][0], base + m["data_offsets"][1])
                   for name, m in header.items() if name != "__metadata__" and select(name))
    merged: list[list[int]] = []
    for lo, hi in spans:
        if merged and lo - merged[-1][1] <= COALESCE_GAP:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    out = []
    for lo, hi in merged:
        for a in range(lo, hi, CHUNK):
            out.append((a, min(a + CHUNK, hi)))
    return out


def stage(repo, out_dir: str, select, link_from: str | None = None, link_globs=(),
          threads: int = 16, small_file_mb: float = 64, log=print) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    t0, fetched, jobs = time.time(), 0, []
    shards = sorted(p for p in repo.files if p.endswith(".safetensors"))
    for path in sorted(repo.files):
        dst = os.path.join(out_dir, path)
        os.makedirs(os.path.dirname(dst) or out_dir, exist_ok=True)
        if link_from and any(fnmatch.fnmatch(path, g) for g in link_globs):
            if not os.path.lexists(dst):
                os.symlink(os.path.join(link_from, path), dst)
            continue
        if path in shards:
            continue
        if repo.files[path] <= small_file_mb * 1e6 and not os.path.exists(dst):
            with open(dst, "wb") as f:
                f.write(repo.read_range(path, 0, repo.files[path]))
    for path in shards:
        dst = os.path.join(out_dir, path)
        if os.path.lexists(dst):
            if os.path.islink(dst):
                continue
        raw, header = header_of(repo, path)
        fd = os.open(dst, os.O_RDWR | os.O_CREAT, 0o644)
        os.ftruncate(fd, repo.files[path])        # sparse: holes cost no disk
        os.pwrite(fd, raw, 0)
        for lo, hi in plan_ranges(header, len(raw), select):
            jobs.append((fd, path, lo, hi))

    def one(job):
        fd, path, lo, hi = job
        os.pwrite(fd, repo.read_range(path, lo, hi), lo)
        return hi - lo

    with ThreadPoolExecutor(threads) as ex:
        for n in ex.map(one, jobs):
            fetched += n
    for fd in {j[0] for j in jobs}:
        os.fsync(fd)
        os.close(fd)
    dt = time.time() - t0
    stats = {"bytes": fetched, "seconds": round(dt, 1), "mb_s": round(fetched / max(dt, 1e-9) / 1e6),
             "requests": len(jobs)}
    log(f"[stage] {fetched/1e9:.2f} GB in {dt:.0f}s ({stats['mb_s']} MB/s, {len(jobs)} requests) -> {out_dir}")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repo", help="Hugging Face repo id, or a local directory with --local")
    ap.add_argument("out_dir")
    ap.add_argument("--local", action="store_true", help="repo is a local directory")
    ap.add_argument("--owner", metavar="K/N", help="routed experts owned by cluster worker K of N")
    ap.add_argument("--include", default=None, metavar="REGEX")
    ap.add_argument("--exclude", default=None, metavar="REGEX")
    ap.add_argument("--link-from", default=None, help="directory (e.g. an HFFS mount) to symlink --link files from")
    ap.add_argument("--link", action="append", default=[], metavar="GLOB")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(argv)
    if not (a.owner or a.include):
        ap.error("give --owner K/N and/or --include REGEX")
    sels = []
    if a.owner:
        sels.append(owner_selector(*parse_owner(a.owner)))
    if a.include:
        sels.append(regex_selector(a.include, a.exclude))
    repo = LocalRepo(a.repo) if a.local else HubRepo(a.repo, workers=a.threads)
    stage(repo, a.out_dir, lambda name: any(f(name) for f in sels), a.link_from, a.link, a.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
