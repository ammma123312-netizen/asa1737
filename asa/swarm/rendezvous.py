"""Rendezvous for an ASA swarm inside one GitHub Actions run.

Free runners cannot accept inbound connections, so each worker exposes its
port through a public TCP relay (bore) and publishes the relay address as a
workflow artifact named `swarm-addr-<i>`. The coordinator polls the run's
artifacts until every worker has reported, then hands the list to Colibrì's
`--cluster-workers`. When it is done it uploads `swarm-done`, which tells the
workers to exit. Only the run's own GITHUB_TOKEN is used.

    python -m asa.swarm.rendezvous wait-workers --count 16 --timeout 1800
    python -m asa.swarm.rendezvous wait-done --timeout 20000
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile

ADDR_PREFIX = "swarm-addr-"
DONE = "swarm-done"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # surface 3xx instead of following it
        return None


_no_redirect = urllib.request.build_opener(_NoRedirect)


def _api(path: str, raw: bool = False):
    """GitHub REST call. Artifact downloads answer 302 to a pre-signed blob URL; that
    URL must be fetched WITHOUT our Authorization header (the blob store rejects it
    with 401), so redirects are followed by hand."""
    base = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    req = urllib.request.Request(base + path, headers={
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    try:
        with _no_redirect.open(req, timeout=60) as r:
            return r.read() if raw else json.load(r)
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308) or not e.headers.get("Location"):
            raise
        with urllib.request.urlopen(e.headers["Location"], timeout=120) as r:
            return r.read() if raw else json.load(r)


def list_artifacts() -> list[dict]:
    repo, run = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_RUN_ID"]
    out, page = [], 1
    while True:
        data = _api(f"/repos/{repo}/actions/runs/{run}/artifacts?per_page=100&page={page}")
        out += data.get("artifacts", [])
        if len(data.get("artifacts", [])) < 100:
            return out
        page += 1


def read_artifact_text(artifact: dict) -> str:
    repo = os.environ["GITHUB_REPOSITORY"]
    blob = _api(f"/repos/{repo}/actions/artifacts/{artifact['id']}/zip", raw=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        return "".join(z.read(n).decode() for n in z.namelist()).strip()


def parse_addresses(texts: list[str]) -> list[str]:
    """Keep well-formed HOST:PORT lines, de-duplicated, in a stable order."""
    seen, out = set(), []
    for t in texts:
        for line in t.splitlines():
            line = line.strip()
            host, _, port = line.rpartition(":")
            if host and port.isdigit() and line not in seen:
                seen.add(line)
                out.append(line)
    return sorted(out)


def parse_worker(text: str) -> dict:
    """A worker's artifact: `relay HOST:PORT` and any number of `direct HOST:PORT` lines.
    A bare HOST:PORT line (older workers) counts as the relay."""
    out: dict = {"relay": None, "direct": []}
    for line in text.splitlines():
        kind, _, addr = line.strip().rpartition(" ")
        if not parse_addresses([addr]):
            continue
        if kind == "direct":
            out["direct"].append(addr)
        elif out["relay"] is None:
            out["relay"] = addr
    return out


def pick_route(worker: dict, reachable) -> tuple[str, str]:
    """Prefer a direct address the coordinator can open (still authenticated by the
    worker's guard), else the relay. Returns (address, 'direct'|'relay')."""
    for addr in worker["direct"]:
        if reachable(addr):
            return addr, "direct"
    return worker["relay"], "relay"


def tcp_reachable(addr: str, timeout: float = 3.0) -> bool:
    import socket
    host, port = addr.rsplit(":", 1)
    try:
        socket.create_connection((host, int(port)), timeout=timeout).close()
        return True
    except OSError:
        return False


def order_by_index(named: list[tuple[str, str]]) -> list[str]:
    """[(artifact name, text)] -> addresses ordered by the worker index in the name.

    Order matters: Colibri gives worker k (position in --cluster-workers) the experts
    with (expert + layer) % n == k, and each worker prefetched exactly that share
    (hffs --warm-owner k/n). Worker i (1-based matrix index) must sit at position i-1."""
    return [w["relay"] for w in workers_by_index(named)]


def workers_by_index(named: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for name, text in named:
        idx = name[len(ADDR_PREFIX):]
        w = parse_worker(text)
        if idx.isdigit() and w["relay"]:
            rows.append((int(idx), w))
    return [w for _, w in sorted(rows, key=lambda r: r[0])]


def wait_workers(count: int, timeout: float, poll: float = 15, direct: bool = False) -> list[str]:
    deadline = time.time() + timeout
    while True:
        arts = [a for a in list_artifacts() if a["name"].startswith(ADDR_PREFIX)]
        workers = workers_by_index([(a["name"], read_artifact_text(a)) for a in arts])
        print(f"[rendezvous] {len(workers)}/{count} workers", file=sys.stderr, flush=True)
        if len(workers) >= count or time.time() > deadline:
            break
        time.sleep(poll)
    if not direct:
        return [w["relay"] for w in workers]
    chosen = [pick_route(w, tcp_reachable) for w in workers]
    for i, (addr, how) in enumerate(chosen):
        print(f"[rendezvous] worker {i + 1}: {how} {addr}", file=sys.stderr, flush=True)
    return [a for a, _ in chosen]


def wait_done(timeout: float, poll: float = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(a["name"] == DONE for a in list_artifacts()):
            return True
        time.sleep(poll)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wait-workers")
    w.add_argument("--count", type=int, required=True)
    w.add_argument("--timeout", type=float, default=1800)
    w.add_argument("--min", type=int, default=1, help="fail if fewer than this reported")
    w.add_argument("--direct", action="store_true", help="use a worker's direct address when reachable")
    d = sub.add_parser("wait-done")
    d.add_argument("--timeout", type=float, default=20000)
    a = ap.parse_args(argv)
    if a.cmd == "wait-workers":
        addrs = wait_workers(a.count, a.timeout, direct=a.direct)
        if len(addrs) < a.min:
            print(f"only {len(addrs)} workers reported", file=sys.stderr)
            return 1
        print(",".join(addrs))
        return 0
    return 0 if wait_done(a.timeout) else 1


if __name__ == "__main__":
    sys.exit(main())
