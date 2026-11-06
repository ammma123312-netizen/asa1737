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
import urllib.request
import zipfile

ADDR_PREFIX = "swarm-addr-"
DONE = "swarm-done"


def _api(path: str, raw: bool = False):
    base = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    req = urllib.request.Request(base + path, headers={
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
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


def wait_workers(count: int, timeout: float, poll: float = 15) -> list[str]:
    deadline = time.time() + timeout
    while True:
        arts = [a for a in list_artifacts() if a["name"].startswith(ADDR_PREFIX)]
        addrs = parse_addresses([read_artifact_text(a) for a in arts])
        print(f"[rendezvous] {len(addrs)}/{count} workers", file=sys.stderr, flush=True)
        if len(addrs) >= count or time.time() > deadline:
            return addrs
        time.sleep(poll)


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
    d = sub.add_parser("wait-done")
    d.add_argument("--timeout", type=float, default=20000)
    a = ap.parse_args(argv)
    if a.cmd == "wait-workers":
        addrs = wait_workers(a.count, a.timeout)
        if len(addrs) < a.min:
            print(f"only {len(addrs)} workers reported", file=sys.stderr)
            return 1
        print(",".join(addrs))
        return 0
    return 0 if wait_done(a.timeout) else 1


if __name__ == "__main__":
    sys.exit(main())
