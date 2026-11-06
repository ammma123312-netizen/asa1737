"""asa/swarm/stage.py: --ram-dir / --ram-budget-gb places shards in RAM up to a budget."""

import json
import os
import struct

import pytest

from asa.swarm import stage as st
from asa.swarm.hffs import owner_selector

N_SHARDS = 4
SEL = owner_selector(0, 2)


def write_safetensors(path, tensors):
    header, off = {}, 0
    for name, blob in tensors.items():
        header[name] = {"dtype": "U8", "shape": [len(blob)], "data_offsets": [off, off + len(blob)]}
        off += len(blob)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(tensors.values()))
    return header, 8 + len(raw)


@pytest.fixture
def src(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "COALESCE_GAP", 0)  # keep holes between tensors
    root = tmp_path / "src"
    root.mkdir()
    shards = {}
    for i in range(1, N_SHARDS + 1):
        name = f"model-{i:05d}-of-{N_SHARDS:05d}.safetensors"
        tensors = {f"layers.{i}.ffn.experts.{e}.w1.weight": bytes([(i * 16 + e) % 255 + 1]) * 10_000
                   for e in range(4)}
        tensors[f"layers.{i}.attn.wq.weight"] = b"\x07" * 5_000
        shards[name] = write_safetensors(root / name, tensors)
    (root / "config.json").write_text("{}")
    return root, shards


def run(src, tmp_path, budget_gb, **kw):
    root, _ = src
    out, ram = tmp_path / "view", tmp_path / "shm"
    logs = []
    stats = st.stage(st.LocalRepo(str(root)), str(out), SEL, log=logs.append,
                     ram_dir=str(ram), ram_budget_gb=budget_gb, **kw)
    return out, ram, stats, logs


def selected_bytes(src):
    """Per shard: header + selected tensors (what stage() counts)."""
    _, shards = src
    return {n: base + sum(m["data_offsets"][1] - m["data_offsets"][0]
                          for t, m in h.items() if SEL(t))
            for n, (h, base) in shards.items()}


def check_view(src, out):
    root, shards = src
    for name, (header, base) in shards.items():
        staged, original = (out / name).read_bytes(), (root / name).read_bytes()
        assert len(staged) == len(original) and staged[:base] == original[:base]
        for t, m in header.items():
            lo, hi = base + m["data_offsets"][0], base + m["data_offsets"][1]
            want = original[lo:hi] if SEL(t) else b"\0" * (hi - lo)
            assert staged[lo:hi] == want, (name, t)
    assert (out / "config.json").read_text() == "{}"


def test_budget_splits_shards(src, tmp_path):
    per = selected_bytes(src)
    names = sorted(per)
    budget = per[names[0]] + per[names[1]]  # exactly two shards fit
    out, ram, stats, logs = run(src, tmp_path, budget / 1e9)
    assert (stats["ram_shards"], stats["disk_shards"]) == (2, 2)
    assert stats["ram_bytes"] == budget and stats["disk_bytes"] == per[names[2]] + per[names[3]]
    for n in names[:2]:
        link = out / n
        assert link.is_symlink() and os.path.realpath(link) == os.path.realpath(ram / n)
        assert (ram / n).is_file() and not (ram / n).is_symlink()
    for n in names[2:]:
        assert (out / n).is_file() and not (out / n).is_symlink() and not (ram / n).exists()
    check_view(src, out)
    assert any(l.startswith("[stage] ram: ") and "in 2 shards, disk: " in l for l in logs)


def test_zero_budget_all_disk(src, tmp_path):
    out, ram, stats, _ = run(src, tmp_path, 0)
    assert (stats["ram_shards"], stats["disk_shards"], stats["ram_bytes"]) == (0, N_SHARDS, 0)
    assert not any(p.is_symlink() for p in out.iterdir()) and not list(ram.iterdir())
    check_view(src, out)


def test_huge_budget_all_ram(src, tmp_path):
    out, ram, stats, _ = run(src, tmp_path, 1000)
    assert (stats["ram_shards"], stats["disk_shards"], stats["disk_bytes"]) == (N_SHARDS, 0, 0)
    assert stats["ram_bytes"] == sum(selected_bytes(src).values())
    assert all((out / n).is_symlink() for n in selected_bytes(src))
    check_view(src, out)


def test_counts_selected_bytes_not_file_size(src, tmp_path):
    per = selected_bytes(src)
    root, _ = src
    one = sorted(per)[0]
    assert per[one] < os.path.getsize(root / one)
    # Budget below the file size but above the selected bytes still puts it in RAM.
    _, _, stats, _ = run(src, tmp_path, (per[one] + 1) / 1e9)
    assert stats["ram_shards"] == 1


def test_restage_moves_between_ram_and_disk(src, tmp_path):
    run(src, tmp_path, 0)
    out, ram, stats, _ = run(src, tmp_path, 1000)            # disk -> RAM
    assert stats["ram_shards"] == N_SHARDS and all(p.is_symlink() for p in out.glob("*.safetensors"))
    check_view(src, out)
    out, ram, stats, _ = run(src, tmp_path, 0)               # RAM -> disk
    assert stats["disk_shards"] == N_SHARDS and not any(p.is_symlink() for p in out.iterdir())
    check_view(src, out)


def test_mount_links_are_left_alone(src, tmp_path):
    root, _ = src
    out, ram, stats, _ = run(src, tmp_path, 1000, link_from=str(root), link_globs=["*-00004-of-*"])
    link = out / "model-00004-of-00004.safetensors"
    assert os.path.realpath(link) == os.path.realpath(root / link.name)
    assert stats["ram_shards"] + stats["disk_shards"] == N_SHARDS - 1


def test_cli_flags(src, tmp_path):
    root, _ = src
    with pytest.raises(SystemExit):
        st.main([str(root), str(tmp_path / "v"), "--local", "--owner", "0/2", "--ram-budget-gb", "1"])
    assert st.main([str(root), str(tmp_path / "v"), "--local", "--owner", "0/2",
                    "--ram-dir", str(tmp_path / "shm"), "--ram-budget-gb", "1"]) == 0
    assert all((tmp_path / "v" / n).is_symlink() for n in selected_bytes(src))
