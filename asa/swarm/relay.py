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

Authentication (mandatory in the swarm): the relay never exposes Colibri's port.
It exposes `guard`, which terminates TLS and requires, as the first line, a GitHub
Actions OIDC token minted for THIS repository and THIS workflow run
(aud=asa-swarm). Only then is the stream spliced to 127.0.0.1:9100; anything else
is dropped before Colibri sees a byte. The coordinator runs `connect`, which opens
one local port per worker and presents its own token. No shared secret exists, so
nothing can leak through public logs or artifacts.

    python -m asa.swarm.relay guard --listen 9200 --target 9100          # worker
    python -m asa.swarm.relay connect --addrs a,b --base-port 19200      # coordinator
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import socket
import ssl
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


def expose_http(port: int, bin_dir: str, log_dir: str) -> str:
    """Public HTTPS URL for a local HTTP service (the API gateway) via a Cloudflare quick
    tunnel. Only for endpoints that enforce their own API key."""
    os.makedirs(log_dir, exist_ok=True)
    cf = fetch_bin("cloudflared", bin_dir)
    log = os.path.join(log_dir, "gateway-cloudflared.log")
    _spawn([cf, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"], log)
    url = _wait_for(log, r"https://[a-z0-9-]+" + re.escape(CF_SUFFIX), 90)
    if not url:
        raise SystemExit("no gateway URL")
    return url


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


# --- authentication ----------------------------------------------------------

OIDC_ISSUER = "https://token.actions.githubusercontent.com"
OIDC_JWKS = OIDC_ISSUER + "/.well-known/jwks"
AUDIENCE = "asa-swarm"


def fetch_oidc_token(audience: str = AUDIENCE) -> str:
    """This job's GitHub Actions OIDC token (needs `permissions: id-token: write`)."""
    url = os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"] + f"&audience={audience}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {os.environ['ACTIONS_ID_TOKEN_REQUEST_TOKEN']}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["value"]


def make_verifier(repository: str, run_id: str, key_for=None, issuer: str = OIDC_ISSUER):
    """Return verify(token) -> claims, raising on anything but this repo's current run.

    key_for(token) gives the signing key; defaults to GitHub's published JWKS."""
    import jwt

    if key_for is None:
        client = jwt.PyJWKClient(OIDC_JWKS, cache_keys=True)
        key_for = lambda tok: client.get_signing_key_from_jwt(tok).key  # noqa: E731

    def verify(token: str) -> dict:
        claims = jwt.decode(token, key_for(token), algorithms=["RS256"], audience=AUDIENCE,
                            issuer=issuer, options={"require": ["exp", "iat", "repository", "run_id"]})
        if claims["repository"] != repository or str(claims["run_id"]) != str(run_id):
            raise PermissionError("token is for another repository or run")
        return claims
    return verify


def self_signed_context(work_dir: str) -> ssl.SSLContext:
    """Server TLS context with a throwaway certificate (confidentiality on the relay hop;
    identity comes from the OIDC token, not the certificate)."""
    import datetime
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "asa-swarm-worker")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=1)).sign(key, hashes.SHA256()))
    os.makedirs(work_dir, exist_ok=True)
    cp, kp = os.path.join(work_dir, "guard.crt"), os.path.join(work_dir, "guard.key")
    with open(cp, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(kp, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()))
    os.chmod(kp, 0o600)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(cp, kp)
    return ctx


async def _pipe(reader, writer):
    try:
        while data := await reader.read(1 << 16):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _splice(r1, w1, r2, w2):
    await asyncio.gather(_pipe(r1, w2), _pipe(r2, w1))


async def guard(listen: int, target: int, verify, tls: ssl.SSLContext | None, log=print) -> None:
    """Worker side: authenticate, then forward to Colibri's local port."""
    async def handle(reader, writer):
        peer = writer.get_extra_info("peername")
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=15)
            if len(line) > 16384:
                raise PermissionError("oversized hello")
            claims = verify(line.decode().strip())
        except Exception as e:  # any failure: drop before Colibri sees a byte
            log(f"[guard] rejected {peer}: {type(e).__name__}")
            writer.close()
            return
        writer.write(b"OK\n")
        await writer.drain()
        log(f"[guard] accepted {peer} job={claims.get('job_workflow_ref', '?')}")
        tr, tw = await asyncio.open_connection("127.0.0.1", target)
        await _splice(reader, writer, tr, tw)

    server = await asyncio.start_server(handle, "127.0.0.1", listen, ssl=tls)
    async with server:
        await server.serve_forever()


async def connect(pairs: list[tuple[int, str]], token_fn, tls: bool = True, log=print) -> None:
    """Coordinator side: each local port forwards to one worker after authenticating."""
    client_ctx = ssl.create_default_context()
    client_ctx.check_hostname = False
    client_ctx.verify_mode = ssl.CERT_NONE  # identity is the OIDC exchange, not the cert

    def handler(host, port):
        async def handle(reader, writer):
            rr, rw = await asyncio.open_connection(host, port, ssl=client_ctx if tls else None,
                                                   server_hostname=host if tls else None)
            rw.write(token_fn().encode() + b"\n")
            await rw.drain()
            if (await asyncio.wait_for(rr.readline(), timeout=15)).strip() != b"OK":
                log(f"[connect] {host}:{port} refused us")
                writer.close()
                return
            await _splice(reader, writer, rr, rw)
        return handle

    servers = []
    for local, addr in pairs:
        host, port = addr.rsplit(":", 1)
        servers.append(await asyncio.start_server(handler(host, int(port)), "127.0.0.1", local))
        log(f"[connect] 127.0.0.1:{local} -> {addr}")
    await asyncio.gather(*(s.serve_forever() for s in servers))


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
    h = sub.add_parser("expose-http")
    h.add_argument("--port", type=int, required=True)
    for p in (h,):
        p.add_argument("--bin-dir", default=os.path.expanduser("~/asa-swarm/bin"))
        p.add_argument("--log-dir", default=os.path.expanduser("~/asa-swarm/relay"))
    g = sub.add_parser("guard")
    g.add_argument("--listen", type=int, default=9200)
    g.add_argument("--target", type=int, default=9100)
    g.add_argument("--work-dir", default=os.path.expanduser("~/asa-swarm/guard"))
    c = sub.add_parser("connect")
    c.add_argument("--addrs", required=True)
    c.add_argument("--base-port", type=int, default=19200)
    pr.add_argument("--addrs", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "guard":
        verify = make_verifier(os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_RUN_ID"])
        asyncio.run(guard(a.listen, a.target, verify, self_signed_context(a.work_dir),
                          log=lambda m: print(m, flush=True)))
        return 0
    if a.cmd == "connect":
        addrs = [x for x in a.addrs.split(",") if x]
        pairs = [(a.base_port + i, addr) for i, addr in enumerate(addrs)]
        print(",".join(f"127.0.0.1:{p}" for p, _ in pairs), flush=True)
        cache: dict = {}

        def token():
            if time.time() - cache.get("t", 0) > 120:
                cache.update(v=fetch_oidc_token(), t=time.time())
            return cache["v"]
        asyncio.run(connect(pairs, token, log=lambda m: print(m, file=sys.stderr, flush=True)))
        return 0
    if a.cmd == "expose-http":
        print(expose_http(a.port, a.bin_dir, a.log_dir))
        return 0
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
