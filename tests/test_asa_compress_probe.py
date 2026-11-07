"""asa/swarm/compress_probe.py: reversible layouts, entropy bound and an offline run."""

import json
import random
import struct

import pytest

pytest.importorskip("zstandard")
pytest.importorskip("lz4")

from asa.swarm import compress_probe as cp


@pytest.mark.parametrize("n", [0, 1, 7, 1000])
def test_layouts_round_trip(n):
    rnd = random.Random(n)
    b = bytes(rnd.randrange(256) for _ in range(n))
    assert cp.nib_join(cp.nib_split(b), n) == b
    assert cp.fp8_join(cp.fp8_split(b), n) == b
    for w in (2, 4):
        bw = b[: n - n % w]
        assert cp.planes_join(cp.planes_split(bw, w), w) == bw


def test_fp8_split_fields():
    # 0b1_0110_101: sign 1, exponent 0110, mantissa 101
    exp, sm = cp.fp8_split(bytes([0b10110101]))
    assert exp[0] & 15 == 0b0110 and sm[0] & 15 == 0b1101


def test_entropy_bound():
    assert cp.entropy_bits(bytes(range(256)) * 4, nibbles=True) == pytest.approx(4.0)
    assert cp.entropy_bits(b"\x11" * 100, nibbles=True) == 0.0
    assert cp.entropy_bits(bytes(range(256)), nibbles=False) == pytest.approx(8.0)


def test_measure_reports_ratios_and_checks_round_trip():
    rnd = random.Random(1)
    uniform = bytes(rnd.randrange(256) for _ in range(50_000))
    r = cp.measure("I8", uniform, min_seconds=0.01)
    assert r["raw/zstd-19"]["ratio"] > 0.99 and "nib-split/lz4" in r  # random = incompressible
    skewed = bytes(rnd.choice([0x7F, 0x7F, 0x7F, 0x80]) for _ in range(50_000))
    r = cp.measure("F8_E8M0", skewed, min_seconds=0.01)
    assert r["raw/zstd-19"]["ratio"] < 0.3 and r["raw/lz4"]["decomp_gb_s"] > 0
    r = cp.measure("F8_E4M3", uniform[:4000], min_seconds=0.01)
    assert "fp8-split/zstd-19" in r
    r = cp.measure("BF16", uniform[:4000], min_seconds=0.01)
    assert "byte-planes/zstd-19" in r


def _write(path, tensors):
    header, off, blobs = {}, 0, []
    for name, (dtype, blob) in tensors.items():
        header[name] = {"dtype": dtype, "shape": [len(blob)], "data_offsets": [off, off + len(blob)]}
        off += len(blob)
        blobs.append(blob)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def test_offline_run(tmp_path, capsys):
    rnd = random.Random(2)
    shard = "model-00001-of-00001.safetensors"
    _write(tmp_path / shard, {
        "layers.3.ffn.experts.7.w1.weight": ("I8", bytes(rnd.randrange(256) for _ in range(8000))),
        "layers.3.ffn.experts.7.w1.scale": ("F8_E8M0", b"\x7f" * 500),
        "layers.3.attn.wq_b.weight": ("F8_E4M3", bytes(rnd.randrange(256) for _ in range(9000))),
    })
    names = ["layers.3.ffn.experts.7.w1.weight", "layers.3.ffn.experts.7.w1.scale",
             "layers.3.attn.wq_b.weight"]
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: shard for n in names}}))
    out = tmp_path / "out.json"
    assert cp.main(["--local", str(tmp_path), "--max-mb", "0.005", "--json", str(out)]) == 0
    d = json.loads(out.read_text())
    assert [r["name"] for r in d["rows"]] == names
    assert d["rows"][2]["bytes"] == 5000 and d["rows"][2]["full_bytes"] == 9000  # sampled prefix
    assert set(d["summary"]) == {"expert-weight", "expert-scale", "dense-weight"}
    assert d["summary"]["expert-scale"]["raw/zstd-19"]["ratio"] < 0.1
