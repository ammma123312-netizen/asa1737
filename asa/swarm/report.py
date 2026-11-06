"""Metrics row for an ASA swarm run, from its `swarm-result` artifact.

    python -m asa.swarm.report 42286888409 [42264619147 ...] [--repo owner/name] [--json]

Downloads the run's `swarm-result` artifact (result.json + engine.err), counts the run's
worker jobs, and prints one Markdown table row per run:

    run | workers | connected | wall_s | tokens | tok/s | load_s | T(1) ms | expert I/O GB | answer

Public repositories need no token (note that unauthenticated API calls are limited to 60 per
hour); set GITHUB_TOKEN (or ASA_GITHUB_TOKEN) for private repositories or higher limits.
Artifact downloads answer 302 to a pre-signed blob URL that rejects our Authorization header,
so the redirect is followed by hand without it (same as rendezvous._api).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import urllib.error
import urllib.request
import zipfile
from typing import Callable, Optional

ARTIFACT = "swarm-result"
DEFAULT_REPO = "ammma123312-netizen/odysseus"

_IO = re.compile(r"expert I/O:[^\n]*?([\d.]+)\s*MB\b")
_T1 = re.compile(r"DSpark gate:\s*T\(1\)\s*([\d.]+)\s*ms")
_CLUSTER = re.compile(r"\[CLUSTER\] coordinator connected to (\d+)")
# Colibri prints e.g. "[v41] 40 layers, ... engram on — loaded in 255.35s".
_LOAD = re.compile(r"loaded in ([0-9.]+)s\b")

COLUMNS = ["run", "workers", "connected", "wall_s", "tokens", "tok/s", "load_s", "T(1) ms",
           "expert I/O GB", "answer"]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _http(url: str, headers: dict):
    """GET with manual redirects: the redirected request carries no Authorization header."""
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(urllib.request.Request(url, headers=headers), timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location") if e.headers else None
        if e.code not in (301, 302, 303, 307, 308) or not loc:
            raise
        with urllib.request.urlopen(loc, timeout=300) as r:  # pre-signed URL: no auth header
            return r.read()


class Api:
    def __init__(self, repo: str, token: Optional[str] = None, fetch: Callable = _http):
        self.repo, self.token, self.fetch = repo, token, fetch
        self.base = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")

    def get(self, path: str, raw: bool = False):
        h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
             "User-Agent": "asa-swarm-report"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        data = self.fetch(f"{self.base}/repos/{self.repo}{path}", h)
        return data if raw else json.loads(data)

    def _paged(self, path: str, key: str) -> list:
        out, page = [], 1
        while True:
            items = self.get(f"{path}?per_page=100&page={page}").get(key, [])
            out += items
            if len(items) < 100:
                return out
            page += 1

    def result_files(self, run: int) -> dict:
        arts = [a for a in self._paged(f"/actions/runs/{run}/artifacts", "artifacts")
                if a.get("name") == ARTIFACT and not a.get("expired")]
        if not arts:
            raise LookupError(f"run {run}: no unexpired '{ARTIFACT}' artifact")
        blob = self.get(f"/actions/artifacts/{arts[-1]['id']}/zip", raw=True)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            return {os.path.basename(n): z.read(n).decode("utf-8", "replace")
                    for n in z.namelist() if not n.endswith("/")}

    def worker_jobs(self, run: int) -> int:
        jobs = self._paged(f"/actions/runs/{run}/jobs", "jobs")
        return sum(1 for j in jobs if str(j.get("name", "")).startswith("worker"))


def parse_engine(text: str) -> dict:
    out = {"expert_io_gb": None, "t1_ms": None, "connected": None, "load_s": None}
    io_mb = [float(m) for m in _IO.findall(text)]
    if io_mb:  # one line per request; the last is the run's total
        out["expert_io_gb"] = round(io_mb[-1] / 1024, 2)
    t1 = _T1.findall(text)
    if t1:
        out["t1_ms"] = float(t1[-1])
    c = _CLUSTER.findall(text)
    if c:
        out["connected"] = int(c[-1])
    load = _LOAD.findall(text)
    if load:
        out["load_s"] = float(load[0])
    return out


def metrics(run: int, files: dict, workers: Optional[int]) -> dict:
    try:
        result = json.loads(files.get("result.json") or "{}")
    except json.JSONDecodeError:
        result = {}
    eng = parse_engine(files.get("engine.err", ""))
    tokens, wall = result.get("completion_tokens"), result.get("wall_s")
    tps = result.get("tok_per_s_end_to_end")
    if tps is None and tokens and wall:
        tps = round(tokens / wall, 3)
    answer = (result.get("answer") or files.get("answer.txt") or "").strip()
    return {"run": run, "workers": workers, "connected": eng["connected"], "wall_s": wall,
            "tokens": tokens, "tok_per_s": tps, "load_s": eng["load_s"], "t1_ms": eng["t1_ms"],
            "expert_io_gb": eng["expert_io_gb"], "cluster_ok": eng["connected"] is not None,
            "answer": answer}


def _cell(v) -> str:
    return "–" if v is None else str(v)


def row(m: dict) -> str:
    ans = " ".join(m["answer"].split())
    ans = (ans[:40] + "…") if len(ans) > 41 else ans
    vals = [m["run"], m["workers"], m["connected"], m["wall_s"], m["tokens"], m["tok_per_s"],
            m["load_s"], m["t1_ms"], m["expert_io_gb"], ans.replace("|", "\\|") or None]
    return "| " + " | ".join(_cell(v) for v in vals) + " |"


def header() -> str:
    return "| " + " | ".join(COLUMNS) + " |\n|" + "---|" * len(COLUMNS)


def report(api: Api, run: int) -> dict:
    files = api.result_files(run)
    try:
        workers = api.worker_jobs(run)
    except Exception:
        workers = None
    return metrics(run, files, workers)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Metrics row for ASA swarm runs.")
    ap.add_argument("runs", nargs="+", type=int)
    ap.add_argument("--repo", default=os.environ.get("ASA_SWARM_REPO")
                    or os.environ.get("GITHUB_REPOSITORY") or DEFAULT_REPO)
    ap.add_argument("--json", action="store_true", help="one JSON object per run")
    args = ap.parse_args(argv)
    api = Api(args.repo, os.environ.get("ASA_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN"))
    if not args.json:
        print(header())
    rc = 0
    for run in args.runs:
        try:
            m = report(api, run)
        except (LookupError, urllib.error.URLError, zipfile.BadZipFile) as e:
            print(f"run {run}: {e}", file=sys.stderr)
            rc = 1
            continue
        print(json.dumps(m, ensure_ascii=False) if args.json else row(m))
    return rc


if __name__ == "__main__":
    sys.exit(main())
