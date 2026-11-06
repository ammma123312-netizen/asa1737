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
