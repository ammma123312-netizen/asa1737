"""Check every external service an ASA swarm node depends on. None needs an account.

    python -m asa.swarm.preflight [--repo deepseek-ai/DeepSeek-V4.1-Flash]

Writes a Markdown table to $GITHUB_STEP_SUMMARY when present. Never fails the
job by itself: a missing service is reported so the next step can fall back.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import socket
import time
import urllib.error
import urllib.request

UA = {"User-Agent": "asa-preflight/1"}


def _timed(fn):
    t = time.time()
    try:
        detail = fn()
        return True, (time.time() - t) * 1000, detail
    except urllib.error.HTTPError as e:  # the service answered: reachable
        return True, (time.time() - t) * 1000, f"reachable (HTTP {e.code})"
    except Exception as e:  # report, never raise
        return False, (time.time() - t) * 1000, f"{type(e).__name__}: {e}"[:120]


def _get(url, headers=None, n=None):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.read(n) if n else r.read()


def check_hf_api(repo):
    _, body = _get(f"https://huggingface.co/api/models/{repo}")
    return f"{json.loads(body).get('id')} reachable"


def check_hf_range(repo):
    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    status, body = _get(url, {"Range": "bytes=0-1023"})
    return f"HTTP {status}, {len(body)} B"


def check_hf_throughput(repo):
    url = f"https://huggingface.co/{repo}/resolve/main/model-00010-of-00048.safetensors"

    def one(i):
        start = 100_000_000 + i * 16_000_000
        return len(_get(url, {"Range": f"bytes={start}-{start + 15_999_999}"})[1])

    t = time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        n = sum(ex.map(one, range(8)))
    return f"{n / (time.time() - t) / 1e6:.0f} MB/s (8 parallel 16 MB ranges)"


def check_tcp(host, port):
    socket.create_connection((host, port), timeout=10).close()
    return f"{host}:{port} open"


def check_github_api():
    token = os.environ.get("GITHUB_TOKEN")
    repo, run = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_RUN_ID")
    if not (token and repo and run):
        return "skipped (not in Actions)"
    _get(f"https://api.github.com/repos/{repo}/actions/runs/{run}/artifacts",
         {"Authorization": f"Bearer {token}"})
    return "artifacts API ok"


def run(repo: str) -> list[tuple[str, bool, float, str]]:
    checks = [
        ("Hugging Face API", lambda: check_hf_api(repo)),
        ("Hugging Face range read", lambda: check_hf_range(repo)),
        ("Hugging Face throughput", lambda: check_hf_throughput(repo)),
        ("GitHub artifacts (rendezvous)", check_github_api),
        ("bore.pub relay (TCP 7835)", lambda: check_tcp("bore.pub", 7835)),
        ("Cloudflare quick tunnel API", lambda: f"HTTP {_get('https://api.trycloudflare.com/', n=1)[0]}"),
        ("GitHub releases (binaries)", lambda: f"HTTP {_get('https://github.com/ekzhang/bore/releases', n=1)[0]}"),
        ("PyPI", lambda: f"HTTP {_get('https://pypi.org/simple/fusepy/', n=1)[0]}"),
    ]
    return [(name, *_timed(fn)) for name, fn in checks]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=os.environ.get("REPO", "deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--role", default=os.environ.get("ASA_ROLE", "node"))
    a = ap.parse_args(argv)
    rows = run(a.repo)
    lines = [f"### ASA preflight ({a.role})", "", "| service | ok | ms | detail |", "|---|---|---|---|"]
    lines += [f"| {n} | {'✅' if ok else '❌'} | {ms:.0f} | {d} |" for n, ok, ms, d in rows]
    text = "\n".join(lines)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
