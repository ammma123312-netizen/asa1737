"""Offline tests for the ASA swarm pieces: HFFS read path and rendezvous parsing."""
import random
import threading

import pytest

pytest.importorskip("requests")

from asa.swarm import hffs
from asa.swarm.rendezvous import parse_addresses


class FakeRepo:
    """Stands in for HubRepo: deterministic bytes, counts range requests."""

    def __init__(self, files):
        self.blobs = {p: bytes(random.Random(p).randrange(256) for _ in range(n)) for p, n in files.items()}
        self.files = {p: len(b) for p, b in self.blobs.items()}
        self.calls = []
        self.lock = threading.Lock()
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(8)

    def read_range(self, path, start, end):
        with self.lock:
            self.calls.append((path, start, end))
        return self.blobs[path][start:end]


@pytest.fixture
def setup(tmp_path):
    repo = FakeRepo({"model-00001-of-00002.safetensors": 300_000,
                     "model-00002-of-00002.safetensors": 50_000, "config.json": 100})
    cache = hffs.BlockCache(str(tmp_path / "c"), max_bytes=10_000_000)
    ops = hffs.build_ops(repo, cache, block=64 * 1024, small_block=1024,
                         small_globs=["*-00002-of-*"], readahead=0)
    return repo, cache, ops


def test_reads_match_source_at_any_offset(setup):
    repo, _, ops = setup
    p = "model-00001-of-00002.safetensors"
    for off, n in [(0, 10), (65_530, 20), (100_000, 150_000), (299_990, 100)]:
        assert ops.read("/" + p, n, off, 0) == repo.blobs[p][off:off + n]


def test_second_read_hits_cache(setup):
    repo, cache, ops = setup
    p = "/model-00001-of-00002.safetensors"
    ops.read(p, 1000, 5000, 0)
    n = len(repo.calls)
    ops.read(p, 1000, 5000, 0)
    assert len(repo.calls) == n and cache.hits >= 1


def test_small_block_files_fetch_tiny_ranges(setup):
    repo, _, ops = setup
    ops.read("/model-00002-of-00002.safetensors", 264, 40_000, 0)
    path, start, end = repo.calls[-1]
    assert end - start <= 1024


def test_listing_and_readonly(setup):
    _, _, ops = setup
    assert "config.json" in ops.readdir("/", 0)
    assert ops.getattr("/config.json")["st_size"] == 100
    with pytest.raises(OSError):
        ops.open("/config.json", 1)  # O_WRONLY


def test_cache_evicts_to_its_budget(tmp_path):
    repo = FakeRepo({"a.bin": 400_000})
    cache = hffs.BlockCache(str(tmp_path / "c"), max_bytes=150_000)
    ops = hffs.build_ops(repo, cache, block=64 * 1024, small_block=1024, small_globs=[], readahead=0)
    ops.read("/a.bin", 400_000, 0, 0)
    assert cache.size <= 150_000


def test_rendezvous_parses_relay_addresses():
    assert parse_addresses(["bore.pub:4001\n", "bore.pub:4000", "junk", "bore.pub:4001"]) == \
        ["bore.pub:4000", "bore.pub:4001"]


def test_rendezvous_orders_workers_by_matrix_index():
    from asa.swarm.rendezvous import order_by_index
    named = [("swarm-addr-10", "bore.pub:1"), ("swarm-addr-2", "bore.pub:9"), ("swarm-addr-1", "bore.pub:5")]
    assert order_by_index(named) == ["bore.pub:5", "bore.pub:9", "bore.pub:1"]


def test_owner_selector_matches_colibri_ownership():
    from asa.swarm.hffs import owner_selector
    n = 18
    owners = {k: owner_selector(k, n) for k in range(n)}
    for layer in (0, 7, 39):
        for e in (0, 5, 383):
            name = f"layers.{layer}.ffn.experts.{e}.w2.weight"
            assert [k for k in range(n) if owners[k](name)] == [(e + layer) % n]
    assert not owners[0]("layers.0.ffn.shared_experts.w1.weight")
    assert not owners[0]("mtp.0.ffn.experts.0.w1.weight")


# --- authenticated relay (guard/connect) -------------------------------------

def _rsa_and_token(repo="owner/asa", run="42", aud="asa-swarm", iss="https://issuer.test"):
    import time as _t
    jwt = pytest.importorskip("jwt")
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = int(_t.time())
    tok = jwt.encode({"repository": repo, "run_id": run, "aud": aud, "iss": iss,
                      "iat": now, "exp": now + 300}, key, algorithm="RS256")
    return key, tok


def test_verifier_accepts_only_this_repo_and_run():
    from asa.swarm.relay import make_verifier
    key, good = _rsa_and_token()
    verify = make_verifier("owner/asa", "42", key_for=lambda t: key.public_key(), issuer="https://issuer.test")
    assert verify(good)["run_id"] == "42"
    for bad in (_rsa_and_token(run="43")[1], _rsa_and_token(repo="evil/x")[1],
                _rsa_and_token(aud="other")[1], "not-a-jwt"):
        with pytest.raises(Exception):
            verify(bad)


def test_guard_and_connect_splice_only_after_auth(tmp_path):
    import asyncio
    from asa.swarm.relay import connect, guard, make_verifier, self_signed_context
    key, good = _rsa_and_token()
    _, wrong_run = _rsa_and_token(run="999")
    verify = make_verifier("owner/asa", "42", key_for=lambda t: key.public_key(), issuer="https://issuer.test")

    async def scenario():
        async def echo(r, w):
            w.write(b"COLIBRI:" + await r.read(5)); await w.drain(); w.close()
        target = await asyncio.start_server(echo, "127.0.0.1", 0)
        tport = target.sockets[0].getsockname()[1]
        gport, good_port, bad_port = 28200, 28201, 28202
        tasks = [asyncio.create_task(guard(gport, tport, verify, self_signed_context(str(tmp_path)), log=lambda m: None)),
                 asyncio.create_task(connect([(good_port, f"127.0.0.1:{gport}")], lambda: good, log=lambda m: None)),
                 asyncio.create_task(connect([(bad_port, f"127.0.0.1:{gport}")], lambda: wrong_run, log=lambda m: None))]
        await asyncio.sleep(0.5)
        r, w = await asyncio.open_connection("127.0.0.1", good_port)
        w.write(b"hello"); await w.drain()
        ok = await asyncio.wait_for(r.read(100), 5)
        r, w = await asyncio.open_connection("127.0.0.1", bad_port)
        w.write(b"hello"); await w.drain()
        refused = await asyncio.wait_for(r.read(100), 5)
        # a raw client that skips TLS/auth entirely must also get nothing from Colibri
        r, w = await asyncio.open_connection("127.0.0.1", gport)
        w.write(b"hello\n"); await w.drain()
        raw = await asyncio.wait_for(r.read(100), 5)
        for t in tasks:
            t.cancel()
        target.close()
        return ok, refused, raw

    ok, refused, raw = asyncio.run(scenario())
    assert ok == b"COLIBRI:hello"
    assert b"COLIBRI" not in refused and b"COLIBRI" not in raw


def test_cap_ranges_keeps_prestaging_inside_budget():
    from asa.swarm.hffs import cap_ranges
    rs = [("a", 0, 4_000_000_000), ("a", 4_000_000_000, 8_000_000_000), ("b", 0, 4_000_000_000)]
    assert cap_ranges(rs, 9) == rs[:2]
    assert cap_ranges(rs, 100) == rs


def test_artifact_download_drops_auth_on_blob_redirect(monkeypatch):
    """GitHub answers artifact downloads with 302 to a blob URL that rejects our token."""
    import http.server
    import io
    import threading
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("addr.txt", "bore.pub:4242\n")
    blob = buf.getvalue()

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/repos/"):
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{srv.server_port}/blob")
                self.end_headers()
            elif self.headers.get("Authorization"):
                self.send_response(401); self.end_headers()
            else:
                self.send_response(200); self.send_header("Content-Length", str(len(blob)))
                self.end_headers(); self.wfile.write(blob)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("GITHUB_API_URL", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    from asa.swarm.rendezvous import read_artifact_text
    assert read_artifact_text({"id": 1}) == "bore.pub:4242"
    srv.shutdown()


# --- sparse local staging (no FUSE in the hot path) -------------------------

def _write_safetensors(path, tensors):
    """Minimal safetensors writer: {name: bytes} -> file; returns header dict."""
    import json as _j
    import struct as _s
    header, off = {}, 0
    for name, blob in tensors.items():
        header[name] = {"dtype": "U8", "shape": [len(blob)], "data_offsets": [off, off + len(blob)]}
        off += len(blob)
    raw = _j.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(_s.pack("<Q", len(raw)) + raw + b"".join(tensors.values()))
    return header, 8 + len(raw)


def test_stage_writes_only_selected_tensors_at_original_offsets(tmp_path, monkeypatch):
    from asa.swarm import stage as st
    from asa.swarm.hffs import owner_selector
    monkeypatch.setattr(st, "COALESCE_GAP", 0)  # keep neighbours separate to check holes
    src = tmp_path / "src"; src.mkdir()
    tensors = {f"layers.{L}.ffn.experts.{E}.w1.weight": bytes([L * 16 + E + 1]) * 4096
               for L in range(2) for E in range(4)}
    tensors["layers.0.attn.wq.weight"] = b"\x07" * 4096
    header, base = _write_safetensors(src / "model-00001-of-00001.safetensors", tensors)
    (src / "config.json").write_text("{}")
    out = tmp_path / "view"
    st.stage(st.LocalRepo(str(src)), str(out), owner_selector(1, 2), log=lambda m: None)
    staged = (out / "model-00001-of-00001.safetensors").read_bytes()
    original = (src / "model-00001-of-00001.safetensors").read_bytes()
    assert len(staged) == len(original) and staged[:base] == original[:base]   # header intact
    for name, m in header.items():
        lo, hi = base + m["data_offsets"][0], base + m["data_offsets"][1]
        if owner_selector(1, 2)(name):
            assert staged[lo:hi] == original[lo:hi], name
        else:
            assert staged[lo:hi] == b"\0" * (hi - lo), name                 # hole
    assert (out / "config.json").read_text() == "{}"


def test_stage_links_files_from_a_mount(tmp_path):
    from asa.swarm import stage as st
    src = tmp_path / "src"; src.mkdir()
    _write_safetensors(src / "model-00001-of-00002.safetensors", {"layers.0.attn.wq.weight": b"a" * 64})
    _write_safetensors(src / "model-00002-of-00002.safetensors", {"engram.table": b"b" * 64})
    out = tmp_path / "view"
    st.stage(st.LocalRepo(str(src)), str(out), lambda n: True, link_from=str(src),
             link_globs=["*-00002-of-*"], log=lambda m: None)
    assert (out / "model-00002-of-00002.safetensors").is_symlink()
    assert not (out / "model-00001-of-00002.safetensors").is_symlink()
