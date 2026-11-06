"""HFFS: mount a Hugging Face model repo as a read-only local directory.

Why: a 763B checkpoint is 510 GB, far more than a free machine's disk. Inference
engines (Colibrì, llama.cpp) only `pread` the bytes they need, so the files do
not have to be local: every read becomes an HTTP Range request to the Hub's CDN.

No account, no key, no cost: public repos are readable anonymously.

What makes it usable rather than toy-slow:
  * the Hub's /resolve/ URL is followed ONCE per file and the signed CDN URL is
    reused, so each read costs one CDN round trip instead of two;
  * reads are cut into blocks, fetched in parallel, de-duplicated (two threads
    asking for the same block share one download) and kept in an LRU disk cache;
  * small-block files: tables read a few hundred bytes at random (DeepSeek-V4.1's
    engram) use tiny blocks so a 264-byte row does not pull megabytes.

Usage:
    python -m asa.swarm.hffs deepseek-ai/DeepSeek-V4.1-Flash /mnt/dsv41 \\
        --cache /var/cache/hffs --cache-gb 20 --small-block '*-0004[78]-*'
    # then build a writable view the engine can use (sidecars, usage files):
    python -m asa.swarm.hffs --make-view /mnt/dsv41 /srv/dsv41-view
"""
from __future__ import annotations

import argparse
import collections
import errno
import fnmatch
import hashlib
import json
import os
import stat
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import requests

HUB = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
SIGNED_URL_TTL = 30 * 60  # refresh signed CDN URLs well before they expire


def _ca_bundle():
    for v in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        if os.environ.get(v):
            return os.environ[v]
    p = "/etc/ssl/certs/ca-certificates.crt"
    return p if os.path.exists(p) else True


class HubRepo:
    """File listing + ranged reads for one repo revision."""

    def __init__(self, repo: str, revision: str = "main", workers: int = 16):
        self.repo, self.revision = repo, revision
        self.verify = _ca_bundle()
        self._local = threading.local()
        self._signed: dict[str, tuple[str, float]] = {}
        self._signed_lock = threading.Lock()
        self.files = self._list()
        self.pool = ThreadPoolExecutor(workers)

    def _session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = self._local.s = requests.Session()
            s.verify = self.verify
            s.headers["User-Agent"] = "asa-hffs/1"
        return s

    def _list(self) -> dict[str, int]:
        url = f"{HUB}/api/models/{self.repo}/tree/{self.revision}?recursive=true"
        files, s = {}, self._session()
        while url:
            r = s.get(url, timeout=60)
            r.raise_for_status()
            for f in r.json():
                if f.get("type") == "file":
                    files[f["path"]] = int(f["size"])
            nxt = r.links.get("next", {}).get("url")
            url = nxt
        return files

    def _signed_url(self, path: str, refresh: bool = False) -> str:
        with self._signed_lock:
            hit = self._signed.get(path)
            if hit and not refresh and time.time() - hit[1] < SIGNED_URL_TTL:
                return hit[0]
        url = f"{HUB}/{self.repo}/resolve/{self.revision}/{urllib.parse.quote(path)}"
        r = self._session().get(url, allow_redirects=False, timeout=60, stream=True)
        r.close()
        loc = r.headers.get("Location") if r.is_redirect else None
        target = urllib.parse.urljoin(url, loc) if loc else url
        with self._signed_lock:
            self._signed[path] = (target, time.time())
        return target

    def read_range(self, path: str, start: int, end: int) -> bytes:
        """Bytes [start, end) of `path`."""
        for attempt in range(4):
            url = self._signed_url(path, refresh=attempt > 0)
            try:
                r = self._session().get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=120)
                if r.status_code in (200, 206):
                    data = r.content
                    if r.status_code == 200:  # server ignored Range
                        data = data[start:end]
                    if len(data) == end - start:
                        return data
                time.sleep(0.5 * (attempt + 1))
            except requests.RequestException:
                time.sleep(0.5 * (attempt + 1))
        raise OSError(errno.EIO, f"range read failed: {path} [{start},{end})")


class BlockCache:
    """Disk LRU of fixed-size blocks with in-flight de-duplication."""

    def __init__(self, root: str, max_bytes: int):
        self.root, self.max = root, max_bytes
        os.makedirs(root, exist_ok=True)
        self.lru: collections.OrderedDict[str, int] = collections.OrderedDict()
        self.size = 0
        self.lock = threading.Lock()
        self.inflight: dict[str, threading.Event] = {}
        for dp, _, fs in os.walk(root):
            for f in fs:
                p = os.path.join(dp, f)
                n = os.path.getsize(p)
                self.lru[p] = n
                self.size += n
        self.hits = self.misses = self.bytes_fetched = 0

    def _key(self, path: str, bs: int, idx: int) -> str:
        h = hashlib.sha1(path.encode()).hexdigest()[:16]
        return os.path.join(self.root, h, f"{bs}-{idx}")

    def get(self, path: str, bs: int, idx: int, fetch) -> bytes:
        key = self._key(path, bs, idx)
        while True:
            with self.lock:
                if key in self.lru:
                    self.lru.move_to_end(key)
                    self.hits += 1
                    try:
                        with open(key, "rb") as fh:
                            return fh.read()
                    except FileNotFoundError:  # evicted between check and open
                        self.size -= self.lru.pop(key, 0)
                ev = self.inflight.get(key)
                if ev is None:
                    ev = self.inflight[key] = threading.Event()
                    owner = True
                else:
                    owner = False
            if not owner:
                ev.wait()
                continue
            try:
                data = fetch()
                os.makedirs(os.path.dirname(key), exist_ok=True)
                tmp = f"{key}.{threading.get_ident()}.tmp"
                with open(tmp, "wb") as fh:
                    fh.write(data)
                os.replace(tmp, key)
                with self.lock:
                    self.misses += 1
                    self.bytes_fetched += len(data)
                    self.lru[key] = len(data)
                    self.size += len(data)
                    while self.size > self.max and len(self.lru) > 1:
                        old, n = self.lru.popitem(last=False)
                        self.size -= n
                        try:
                            os.remove(old)
                        except FileNotFoundError:
                            pass
                return data
            finally:
                with self.lock:
                    self.inflight.pop(key, None)
                ev.set()


def build_ops(repo: HubRepo, cache: BlockCache, block: int, small_block: int, small_globs: list[str],
              readahead: int = 8):
    try:
        from fuse import FuseOSError, Operations
    except ImportError:  # unit tests exercise the read path without libfuse
        Operations = object

        class FuseOSError(OSError):
            def __init__(self, err):
                super().__init__(err, os.strerror(err))

    dirs: set[str] = {""}
    for p in repo.files:
        parts = p.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    now = time.time()

    def bs_for(path: str) -> int:
        return small_block if any(fnmatch.fnmatch(path, g) for g in small_globs) else block

    # Readahead: a sequential reader (an engine loading dense weights) would
    # otherwise pay one CDN round trip per block. When a file's reads continue
    # where the previous one ended, fetch the next `readahead` blocks in the
    # background so they are cached by the time the reader gets there.
    ra_pool = ThreadPoolExecutor(max(4, readahead * 2))
    last_end: dict[str, int] = {}
    queued: set[tuple[str, int]] = set()
    ra_lock = threading.Lock()

    def prefetch(p: str, bs: int, total: int, start_idx: int):
        n_blocks = (total + bs - 1) // bs
        for i in range(start_idx, min(start_idx + readahead, n_blocks)):
            with ra_lock:
                if (p, i) in queued:
                    continue
                queued.add((p, i))

            def job(i=i):
                try:
                    cache.get(p, bs, i, lambda: repo.read_range(p, i * bs, min((i + 1) * bs, total)))
                except OSError:
                    pass
                finally:
                    with ra_lock:
                        queued.discard((p, i))
            ra_pool.submit(job)

    class Ops(Operations):
        def getattr(self, path, fh=None):
            p = path.lstrip("/")
            if p in dirs:
                return dict(st_mode=stat.S_IFDIR | 0o555, st_nlink=2, st_size=0,
                            st_mtime=now, st_atime=now, st_ctime=now)
            if p in repo.files:
                return dict(st_mode=stat.S_IFREG | 0o444, st_nlink=1, st_size=repo.files[p],
                            st_mtime=now, st_atime=now, st_ctime=now)
            raise FuseOSError(errno.ENOENT)

        def readdir(self, path, fh):
            p = path.lstrip("/")
            pre = f"{p}/" if p else ""
            names = {".", ".."}
            for f in list(repo.files) + list(dirs):
                if f and f.startswith(pre):
                    names.add(f[len(pre):].split("/")[0])
            return sorted(n for n in names if n)

        def open(self, path, flags):
            if flags & (os.O_WRONLY | os.O_RDWR):
                raise FuseOSError(errno.EROFS)
            if path.lstrip("/") not in repo.files:
                raise FuseOSError(errno.ENOENT)
            return 0

        def read(self, path, size, offset, fh):
            p = path.lstrip("/")
            total = repo.files[p]
            end = min(offset + size, total)
            if offset >= end:
                return b""
            bs = bs_for(p)
            first, last = offset // bs, (end - 1) // bs

            def one(i):
                return cache.get(p, bs, i, lambda: repo.read_range(p, i * bs, min((i + 1) * bs, total)))

            if first == last:
                blocks = [one(first)]
            else:
                blocks = list(repo.pool.map(one, range(first, last + 1)))
            if readahead and bs == block:
                with ra_lock:
                    sequential = abs(last_end.get(p, -1) - offset) <= 4 * bs
                    last_end[p] = end
                if sequential:
                    prefetch(p, bs, total, last + 1)
            buf = b"".join(blocks)
            lo = offset - first * bs
            return buf[lo:lo + (end - offset)]

        def statfs(self, path):
            return dict(f_bsize=4096, f_blocks=sum(repo.files.values()) // 4096, f_bavail=0, f_bfree=0)

    return Ops()


def safetensors_ranges(repo: HubRepo, include, exclude) -> list[tuple[str, int, int]]:
    """Byte ranges of tensors whose names match `include` and not `exclude` (regexes)."""
    import re
    import struct
    inc, exc = re.compile(include), re.compile(exclude) if exclude else None
    out = []
    for path in sorted(p for p in repo.files if p.endswith(".safetensors")):
        n = struct.unpack("<Q", repo.read_range(path, 0, 8))[0]
        header = json.loads(repo.read_range(path, 8, 8 + n))
        base = 8 + n
        for name, meta in header.items():
            if name == "__metadata__" or not inc.search(name) or (exc and exc.search(name)):
                continue
            lo, hi = meta["data_offsets"]
            out.append((path, base + lo, base + hi))
    return out


def warm(repo: HubRepo, cache: BlockCache, block: int, ranges, threads: int = 32) -> int:
    """Fetch the blocks covering `ranges` into the cache, in parallel. Returns bytes."""
    blocks = sorted({(p, i) for p, lo, hi in ranges for i in range(lo // block, (hi - 1) // block + 1)})
    total = 0

    def one(pi):
        p, i = pi
        size = repo.files[p]
        return len(cache.get(p, block, i, lambda: repo.read_range(p, i * block, min((i + 1) * block, size))))

    with ThreadPoolExecutor(threads) as ex:
        for n in ex.map(one, blocks):
            total += n
    return total


def make_view(mount: str, view: str) -> int:
    """Writable directory of symlinks into the mount, so engines can write sidecars."""
    n = 0
    for dp, _, fs in os.walk(mount):
        for f in fs:
            src = os.path.join(dp, f)
            dst = os.path.join(view, os.path.relpath(src, mount))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.lexists(dst):
                os.symlink(src, dst)
                n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repo", nargs="?")
    ap.add_argument("mountpoint", nargs="?")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/asa-hffs"))
    ap.add_argument("--cache-gb", type=float, default=20)
    ap.add_argument("--block-mb", type=float, default=4)
    ap.add_argument("--small-block-kb", type=float, default=16)
    ap.add_argument("--small-block", action="append", default=[], metavar="GLOB",
                    help="files read in tiny random pieces (e.g. engram tables)")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--readahead", type=int, default=8, help="blocks prefetched for sequential readers")
    ap.add_argument("--warm-include", default=None, metavar="REGEX",
                    help="prefetch safetensors tensors matching this at mount time")
    ap.add_argument("--warm-exclude", default=None, metavar="REGEX")
    ap.add_argument("--make-view", nargs=2, metavar=("MOUNT", "VIEW"))
    ap.add_argument("--list", action="store_true", help="print files and sizes, then exit")
    a = ap.parse_args(argv)

    if a.make_view:
        print(f"linked {make_view(*a.make_view)} files")
        return 0
    if not a.repo:
        ap.error("repo is required")
    repo = HubRepo(a.repo, a.revision, a.threads)
    if a.list:
        print(json.dumps(repo.files, indent=1))
        return 0
    if not a.mountpoint:
        ap.error("mountpoint is required")
    from fuse import FUSE

    os.makedirs(a.mountpoint, exist_ok=True)
    cache = BlockCache(os.path.join(a.cache, a.repo.replace("/", "__")), int(a.cache_gb * 1e9))
    ops = build_ops(repo, cache, int(a.block_mb * 2**20), int(a.small_block_kb * 1024), a.small_block,
                    a.readahead)
    print(f"[hffs] {a.repo}: {len(repo.files)} files, {sum(repo.files.values())/1e9:.1f} GB "
          f"-> {a.mountpoint} (cache {a.cache_gb} GB)", flush=True)

    def stats():
        while True:
            time.sleep(30)
            print(f"[hffs] hits={cache.hits} misses={cache.misses} "
                  f"fetched={cache.bytes_fetched/1e9:.2f} GB cache={cache.size/1e9:.2f} GB", flush=True)

    threading.Thread(target=stats, daemon=True).start()

    if a.warm_include:
        def do_warm():
            t = time.time()
            rs = safetensors_ranges(repo, a.warm_include, a.warm_exclude)
            want = sum(hi - lo for _, lo, hi in rs)
            print(f"[hffs] warming {len(rs)} tensors, {want/1e9:.2f} GB", flush=True)
            got = warm(repo, cache, int(a.block_mb * 2**20), rs, a.threads)
            print(f"[hffs] warm done: {got/1e9:.2f} GB of blocks in {time.time()-t:.0f}s "
                  f"({got/max(time.time()-t, 1e-9)/1e6:.0f} MB/s)", flush=True)
        threading.Thread(target=do_warm, daemon=True).start()
    # kernel_cache: re-reads are served from the page cache without entering Python.
    # max_read: fewer, larger FUSE requests per expert read.
    FUSE(ops, a.mountpoint, foreground=True, nothreads=False, ro=True,
         kernel_cache=True, max_read=1 << 20, attr_timeout=3600, entry_timeout=3600)
    return 0


if __name__ == "__main__":
    sys.exit(main())
