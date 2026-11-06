"""Expose a swarm worker's TCP port to the coordinator, with no account anywhere.

Free runners cannot accept inbound connections, so a worker needs a public relay:

  1. bore (ekzhang/bore) via the free public server bore.pub: raw TCP, lowest
     overhead. Address published as `bore.pub:<port>`.
  2. Fallback: a Cloudflare *quick* tunnel (trycloudflare.com, no account) that
     carries the TCP stream inside a WebSocket (websocat on both ends).
     Address published as `<name>.trycloudflare.com:443`; the coordinator runs
     `bridge`, which turns each such address into a local 127.0.0.1 port.

    python -m asa.swarm.relay expose --port 9100            # prints the address
    python -m asa.swarm.relay bridge --addrs a,b,c           # prints usable list
    python -m asa.swarm.relay probe --addrs a,b,c            # connect latency
"""
from __future__ import annotations

import argparse
import os
import re
import socket
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request

BORE_VERSION = os.environ.get("BORE_VERSION", "v0.6.0")
WEBSOCAT_VERSION = os.environ.get("WEBSOCAT_VERSION", "v1.14.0")
BINS = {
    "bore": f"https://github.com/ekzhang/bore/releases/download/{BORE_VERSION}/"
            f"bore-{BORE_VERSION}-x86_64-unknown-linux-musl.tar.gz",
    "websocat": f"https://github.com/vi/websocat/releases/download/{WEBSOCAT_VERSION}/"
                "websocat.x86_64-unknown-linux-musl",
    "cloudflared": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
}
CF_SUFFIX = ".trycloudflare.com"


def fetch_bin(name: str, bin_dir: str) -> str:
    path = os.path.join(bin_dir, name)
    if os.path.exists(path):
        return path
    os.makedirs(bin_dir, exist_ok=True)
    url = BINS[name]
    data = urllib.request.urlopen(url, timeout=120).read()
    if url.endswith(".tar.gz"):
        tmp = path + ".tgz"
        with open(tmp, "wb") as f:
            f.write(data)
        with tarfile.open(tmp) as t:
            t.extract(name, bin_dir, filter="data")
        os.remove(tmp)
    else:
        with open(path, "wb") as f:
            f.write(data)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


def _spawn(cmd: list[str], log: str) -> subprocess.Popen:
    """Start a long-lived helper that outlives this Python process."""
    return subprocess.Popen(cmd, stdout=open(log, "ab"), stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def _wait_for(log: str, pattern: str, timeout: float) -> str | None:
    rx, deadline = re.compile(pattern), time.time() + timeout
    while time.time() < deadline:
        try:
            m = rx.search(open(log, errors="replace").read())
            if m:
                return m.group(0)
        except FileNotFoundError:
            pass
        time.sleep(1)
    return None


def expose_bore(port: int, host: str, bin_dir: str, log_dir: str) -> str | None:
    bore = fetch_bin("bore", bin_dir)
    log = os.path.join(log_dir, "bore.log")
    _spawn([bore, "local", str(port), "--to", host], log)
    return _wait_for(log, re.escape(host) + r":\d+", 30)


def expose_cloudflare(port: int, bin_dir: str, log_dir: str, ws_port: int = 18081) -> str | None:
    ws, cf = fetch_bin("websocat", bin_dir), fetch_bin("cloudflared", bin_dir)
    _spawn([ws, "--binary", "--exit-on-eof", "-E", f"ws-l:127.0.0.1:{ws_port}", f"tcp:127.0.0.1:{port}"],
           os.path.join(log_dir, "websocat-worker.log"))
    log = os.path.join(log_dir, "cloudflared.log")
    _spawn([cf, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{ws_port}"], log)
    url = _wait_for(log, r"https://[a-z0-9-]+" + re.escape(CF_SUFFIX), 60)
    return f"{url[len('https://'):]}:443" if url else None


def expose(port: int, prefer: list[str], bore_host: str, bin_dir: str, log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    for kind in prefer:
        try:
            addr = (expose_bore(port, bore_host, bin_dir, log_dir) if kind == "bore"
                    else expose_cloudflare(port, bin_dir, log_dir))
        except Exception as e:  # a relay being down must not stop the next one
            print(f"[relay] {kind} failed: {e}", file=sys.stderr)
            addr = None
        if addr:
            print(f"[relay] {kind}: {addr}", file=sys.stderr)
            return addr
        print(f"[relay] {kind}: no address, trying next", file=sys.stderr)
    raise SystemExit("no relay worked")


def bridge(addrs: list[str], bin_dir: str, log_dir: str, base_port: int = 19100) -> list[str]:
    """Coordinator side: Cloudflare addresses become local ports; bore ones pass through."""
    out, ws = [], None
    for i, a in enumerate(addrs):
        host = a.rsplit(":", 1)[0]
        if not host.endswith(CF_SUFFIX):
            out.append(a)
            continue
        ws = ws or fetch_bin("websocat", bin_dir)
        port = base_port + i
        _spawn([ws, "--binary", "-E", f"tcp-l:127.0.0.1:{port}", f"wss://{host}/"],
               os.path.join(log_dir, f"websocat-{i}.log"))
        out.append(f"127.0.0.1:{port}")
    time.sleep(2 if ws else 0)
    return out


def probe(addrs: list[str], timeout: float = 10) -> list[tuple[str, float | None]]:
    rows = []
    for a in addrs:
        host, port = a.rsplit(":", 1)
        t = time.time()
        try:
            socket.create_connection((host, int(port)), timeout=timeout).close()
            rows.append((a, (time.time() - t) * 1000))
        except OSError:
            rows.append((a, None))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("expose")
    e.add_argument("--port", type=int, required=True)
    e.add_argument("--prefer", default=os.environ.get("ASA_RELAY", "bore,cloudflare"))
    e.add_argument("--bore-host", default=os.environ.get("ASA_BORE_HOST", "bore.pub"))
    for p in (e, b := sub.add_parser("bridge"), pr := sub.add_parser("probe")):
        p.add_argument("--bin-dir", default=os.path.expanduser("~/asa-swarm/bin"))
        p.add_argument("--log-dir", default=os.path.expanduser("~/asa-swarm/relay"))
    b.add_argument("--addrs", required=True)
    pr.add_argument("--addrs", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "expose":
        print(expose(a.port, a.prefer.split(","), a.bore_host, a.bin_dir, a.log_dir))
    elif a.cmd == "bridge":
        os.makedirs(a.log_dir, exist_ok=True)
        print(",".join(bridge([x for x in a.addrs.split(",") if x], a.bin_dir, a.log_dir)))
    else:
        for addr, ms in probe([x for x in a.addrs.split(",") if x]):
            print(f"{addr}: {'unreachable' if ms is None else f'{ms:.0f} ms'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
