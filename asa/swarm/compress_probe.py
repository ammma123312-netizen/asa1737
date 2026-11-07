"""Measure lossless compressibility of real checkpoint tensors (ASA: "experts in RAM, compressed?").

    python -m asa.swarm.compress_probe [--repo deepseek-ai/DeepSeek-V4.1-Flash] [--max-mb 16] [--json out.json]

Reads tensors straight from the Hub with HTTP Range requests (safetensors header first), so
nothing but the sampled bytes is downloaded. For each tensor it tries several reversible
layouts and two codecs, and checks every round trip byte for byte:

  layouts  raw          the bytes as stored
           nib-split    FP4 packed 2/byte (I8): low nibbles and high nibbles as two packed
                        streams (ZipNN-style grouping of like fields)
           fp8-split    FP8 e4m3: 4 exponent bits and sign+3 mantissa bits as two packed
                        nibble streams
           byte-planes  multi-byte dtypes (F32/BF16/F16): byte k of every element together
  codecs   zstd -19, lz4 (default frame)

and reports compressed/original ratio, decompression GB/s on this machine (of original
bytes), and the order-0 entropy bound (bits per nibble for 4-bit data, per byte otherwise),
which caps what any entropy coder without context modelling can reach.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
import urllib.request
from collections import Counter
from typing import Callable

DEFAULT_REPO = "deepseek-ai/DeepSeek-V4.1-Flash"

# (layer, expert, proj) samples spread over the stack, plus dense FP8 attention weights.
EXPERT_SAMPLES = [(3, 7, "w1"), (3, 7, "w2"), (3, 7, "w3"), (10, 120, "w1"), (17, 300, "w2"),
                  (24, 5, "w3"), (30, 200, "w1"), (35, 77, "w2"), (39, 383, "w3"), (20, 64, "w1")]
DENSE_SAMPLES = ["layers.3.attn.wq_b.weight", "layers.3.attn.wkv.weight",
                 "layers.20.attn.wo_a.weight", "layers.20.attn.wo_b.weight",
                 "layers.38.attn.wq_a.weight"]


# ---------------------------------------------------------------- reversible layouts
def _pack_nibbles(vals: bytes) -> bytes:
    """vals: one nibble (0..15) per byte -> two per byte (pads one 0 nibble if odd)."""
    if len(vals) % 2:
        vals += b"\0"
    lo, hi = vals[0::2], vals[1::2]
    return bytes(a | (b << 4) for a, b in zip(lo, hi))


def _unpack_nibbles(packed: bytes, n: int) -> bytes:
    out = bytearray(2 * len(packed))
    out[0::2] = bytes(b & 15 for b in packed)
    out[1::2] = bytes(b >> 4 for b in packed)
    return bytes(out[:n])


def nib_split(b: bytes) -> list[bytes]:
    return [_pack_nibbles(bytes(x & 15 for x in b)), _pack_nibbles(bytes(x >> 4 for x in b))]


def nib_join(parts: list[bytes], n: int) -> bytes:
    lo, hi = _unpack_nibbles(parts[0], n), _unpack_nibbles(parts[1], n)
    return bytes(a | (h << 4) for a, h in zip(lo, hi))


def fp8_split(b: bytes) -> list[bytes]:
    exp = bytes((x >> 3) & 15 for x in b)                  # e4m3: s eeee mmm
    sm = bytes(((x >> 7) << 3) | (x & 7) for x in b)
    return [_pack_nibbles(exp), _pack_nibbles(sm)]


def fp8_join(parts: list[bytes], n: int) -> bytes:
    e, sm = _unpack_nibbles(parts[0], n), _unpack_nibbles(parts[1], n)
    return bytes(((s >> 3) << 7) | (x << 3) | (s & 7) for x, s in zip(e, sm))


def planes_split(b: bytes, width: int) -> list[bytes]:
    return [b[k::width] for k in range(width)]


def planes_join(parts: list[bytes], width: int) -> bytes:
    out = bytearray(sum(map(len, parts)))
    for k, p in enumerate(parts):
        out[k::width] = p
    return bytes(out)


DTYPE_WIDTH = {"F32": 4, "BF16": 2, "F16": 2, "I32": 4, "I16": 2}
FP8 = {"F8_E4M3", "F8_E4M3FN"}


def layouts(dtype: str, b: bytes) -> dict[str, tuple[list[bytes], Callable[[list[bytes]], bytes]]]:
    n = len(b)
    out = {"raw": ([b], lambda p: p[0])}
    if dtype in ("I8", "U8"):  # FP4 x2 packed
        out["nib-split"] = (nib_split(b), lambda p: nib_join(p, n))
    if dtype in FP8:
        out["fp8-split"] = (fp8_split(b), lambda p: fp8_join(p, n))
    if dtype in DTYPE_WIDTH:
        w = DTYPE_WIDTH[dtype]
        out["byte-planes"] = (planes_split(b, w), lambda p: planes_join(p, w))
    return out


def entropy_bits(b: bytes, nibbles: bool) -> float:
    """Order-0 Shannon entropy in bits per symbol (nibble or byte)."""
    syms = [x & 15 for x in b] + [x >> 4 for x in b] if nibbles else b
    c, n = Counter(syms), len(syms)
    return -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0.0


# ---------------------------------------------------------------- codecs
def codecs():
    import lz4.frame
    import zstandard

    zc, zd = zstandard.ZstdCompressor(level=19), zstandard.ZstdDecompressor()
    return {"zstd-19": (zc.compress, zd.decompress),
            "lz4": (lz4.frame.compress, lz4.frame.decompress)}


def measure(dtype: str, data: bytes, min_seconds: float = 0.2) -> dict:
    res = {"bytes": len(data), "dtype": dtype,
           "entropy_bits": round(entropy_bits(data, dtype in ("I8", "U8") or dtype in FP8), 3)}
    for lname, (parts, join) in layouts(dtype, data).items():
        for cname, (comp, decomp) in codecs().items():
            blobs = [comp(p) for p in parts]
            out = join([decomp(x) for x in blobs])
            if out != data:
                raise AssertionError(f"round trip failed: {lname}/{cname}")
            reps, t0 = 0, time.perf_counter()
            while True:  # time decompression + re-layout of the whole tensor
                join([decomp(x) for x in blobs]) if lname != "raw" else decomp(blobs[0])
                reps += 1
                dt = time.perf_counter() - t0
                if dt >= min_seconds:
                    break
            res[f"{lname}/{cname}"] = {
                "ratio": round(sum(map(len, blobs)) / len(data), 4),
                "decomp_gb_s": round(len(data) * reps / dt / 1e9, 3),
            }
    return res


# ---------------------------------------------------------------- reading
class HubReader:
    def __init__(self, repo: str, revision: str = "main"):
        self.base = f"https://huggingface.co/{repo}/resolve/{revision}/"
        self._headers: dict[str, tuple[int, dict]] = {}

    def range(self, path: str, lo: int, hi: int) -> bytes:
        req = urllib.request.Request(self.base + path, headers={"Range": f"bytes={lo}-{hi - 1}",
                                                               "User-Agent": "asa-compress-probe"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        if len(data) != hi - lo:
            raise IOError(f"{path}: wanted {hi - lo} bytes, got {len(data)}")
        return data

    def json(self, path: str):
        with urllib.request.urlopen(self.base + path, timeout=120) as r:
            return json.load(r)

    def header(self, path: str) -> tuple[int, dict]:
        if path not in self._headers:
            n = struct.unpack("<Q", self.range(path, 0, 8))[0]
            self._headers[path] = (8 + n, json.loads(self.range(path, 8, 8 + n)))
        return self._headers[path]


class LocalReader(HubReader):
    """Same interface over a local directory (tests)."""

    def __init__(self, root: str):
        self.root, self._headers = root, {}

    def range(self, path, lo, hi):
        with open(os.path.join(self.root, path), "rb") as f:
            f.seek(lo)
            return f.read(hi - lo)

    def json(self, path):
        with open(os.path.join(self.root, path)) as f:
            return json.load(f)


def tensor_bytes(reader, weight_map: dict, name: str, max_bytes: int) -> tuple[str, bytes, int]:
    path = weight_map[name]
    base, header = reader.header(path)
    m = header[name]
    lo, hi = base + m["data_offsets"][0], base + m["data_offsets"][1]
    return m["dtype"], reader.range(path, lo, min(hi, lo + max_bytes)), hi - lo


def sample_names(weight_map: dict) -> list[str]:
    names = []
    for L, E, w in EXPERT_SAMPLES:
        for suffix in ("weight", "scale"):
            n = f"layers.{L}.ffn.experts.{E}.{w}.{suffix}"
            if n in weight_map:
                names.append(n)
    return names + [n for n in DENSE_SAMPLES if n in weight_map]


def summarize(rows: list[dict]) -> dict:
    """Byte-weighted ratio per (group, variant)."""
    out: dict = {}
    for r in rows:
        group = "expert" if ".experts." in r["name"] else "dense"
        group += "-scale" if r["name"].endswith(".scale") else "-weight"
        g = out.setdefault(group, {"bytes": 0})
        g["bytes"] += r["bytes"]
        for k, v in r.items():
            if isinstance(v, dict) and "ratio" in v:
                acc = g.setdefault(k, [0.0, 0.0, 0])
                acc[0] += v["ratio"] * r["bytes"]
                acc[1] += v["decomp_gb_s"]
                acc[2] += 1
    for g in out.values():
        for k, v in list(g.items()):
            if isinstance(v, list):
                g[k] = {"ratio": round(v[0] / g["bytes"], 4), "decomp_gb_s": round(v[1] / v[2], 3)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Lossless compressibility of checkpoint tensors.")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--local", help="read from a local directory instead of the Hub")
    ap.add_argument("--max-mb", type=float, default=16, help="bytes sampled per tensor (from its start)")
    ap.add_argument("--json", help="write all rows + summary here")
    a = ap.parse_args(argv)
    reader = LocalReader(a.local) if a.local else HubReader(a.repo)
    wm = reader.json("model.safetensors.index.json")["weight_map"]
    rows = []
    for name in sample_names(wm):
        dtype, data, full = tensor_bytes(reader, wm, name, int(a.max_mb * 1e6))
        r = {"name": name, "full_bytes": full, **measure(dtype, data)}
        rows.append(r)
        best = min((k for k in r if "/" in k), key=lambda k: r[k]["ratio"])
        print(f"{name:42s} {dtype:9s} {len(data)/1e6:7.2f} MB  H={r['entropy_bits']:.3f}  "
              f"raw/zstd {r['raw/zstd-19']['ratio']:.3f}  raw/lz4 {r['raw/lz4']['ratio']:.3f}  "
              f"best {best} {r[best]['ratio']:.3f} @ {r[best]['decomp_gb_s']:.2f} GB/s", flush=True)
    summary = summarize(rows)
    print(json.dumps(summary, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"repo": a.repo, "max_mb": a.max_mb, "rows": rows, "summary": summary}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
