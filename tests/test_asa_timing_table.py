"""asa/swarm/timing_table.py on a fixture in the exact formats relay.py, rendezvous.py and
Colibri print, with A9's worker-1 numbers (guard 1.7 s / 95 calls, connect 6.8 s / 89 calls)."""

import json

import pytest

from asa.swarm import timing_table as tt


def timing(label, n, total, mean=None):
    s = {"n": n, "mean_ms": mean or round(1000 * total / n, 1), "p50_ms": 1.0, "p95_ms": 2.0,
         "total_s": total}
    return f"[timing] {label} {json.dumps(s)}"


@pytest.fixture
def run(tmp_path):
    res = tmp_path / "swarm-result"
    (res / "asa-swarm").mkdir(parents=True)
    (res / "result.json").write_text(json.dumps({
        "answer": "92", "wall_s": 410.0, "completion_tokens": 8,
        "tokens": [{"t": 100.0, "text": "There"}, {"t": 150.0, "text": " are"},
                   {"t": 190.0, "text": " 92"}, {"t": 250.0, "text": " solutions"},
                   {"t": 300.0, "text": "."}, {"t": 330.0, "text": " The"},
                   {"t": 380.0, "text": " answer"}, {"t": 410.0, "text": " is"}]}))
    (res / "engine.err").write_text(
        "[v41] 40 layers, 384 experts/layer (6 routed), dim 5120, hc 4, window 128, "
        "engram on — loaded in 255.35s\n"
        "[CLUSTER] coordinator connected to 3 expert worker(s)\n"
        "[v41] expert I/O: 78 misses, 1466.4 MB in 5.24s (0.28 GB/s); prefill 0.0 MB, disk 0.00s, "
        "matmul 0.00s, wall 467.68s\n"
        "[v41] DSpark: 3 of 6 drafts accepted this turn (8 written)\n"
        "[v41] DSpark gate: T(1) 49000.0 ms, rows 3:1.07 4:1.49, rho 0.23; 2 rounds took a plain step\n")
    (res / "rendezvous.log").write_text(
        "[rendezvous] 2/3 workers\n[rendezvous] 3/3 workers\n"
        "[rendezvous] worker 1: relay bore.pub:12227\n"
        "[rendezvous] worker 2: relay bore.pub:40001\n"
        "[rendezvous] worker 3: relay bore.pub:40002\n")
    (res / "asa-swarm" / "connect.log").write_text("\n".join([
        "[connect] 127.0.0.1:9301 -> bore.pub:12227",
        "[connect] 127.0.0.1:9302 -> bore.pub:40001",
        "[connect] 127.0.0.1:9303 -> bore.pub:40002",
        "[connect] authenticated to bore.pub:12227",
        "[connect] authenticated to bore.pub:40001",
        timing("connect bore.pub:12227", 40, 3.0),         # earlier periodic report
        "[pipe] connect bore.pub:40001 <- closed by peer after 1000 B",
        "[connect] authenticated to bore.pub:40001",       # dropped and reconnected
        timing("connect bore.pub:12227", 89, 6.8),         # last report wins
        timing("connect bore.pub:40001", 80, 12.4),
        "[pipe] connect bore.pub:40002 -> error after 0 B: TimeoutError: timed out",
    ]) + "\n")
    for k, (n, total) in {1: (95, 1.7), 2: (90, 2.4)}.items():
        w = tmp_path / f"worker-logs-{k}"
        w.mkdir()
        (w / "guard.log").write_text("\n".join([
            "[guard] rejected ('10.0.0.9', 1111): PermissionError",
            "[guard] accepted ('10.1.0.4', 5555) job=o/r/.github/workflows/asa-swarm.yml@refs/heads/x",
            timing("guard ('10.1.0.4', 5555)", 3, 0.1),     # a short probe connection
            timing("guard ('10.1.0.4', 6666)", n, total),
        ]) + "\n")
        (w / "worker.log").write_text("[CLUSTER] worker ready\n" + ("Killed\n" if k == 2 else ""))
        (w / "bore.log").write_text("listening at bore.pub\n")
    w3 = tmp_path / "worker-logs-3"
    w3.mkdir()
    (w3 / "guard.log").write_text("")  # never accepted
    steps = {"total_count": 3, "jobs": [
        {"name": "coordinator", "steps": [
            {"name": "Node setup", "started_at": "2026-11-07T02:37:30Z", "completed_at": "2026-11-07T02:40:30Z"},
            {"name": "Run the requested task", "started_at": "2026-11-07T02:48:00Z", "completed_at": "2026-11-07T02:58:30Z"}]},
        {"name": "worker (1)", "steps": [
            {"name": "Node setup", "started_at": "2026-11-07T02:37:30Z", "completed_at": "2026-11-07T02:39:30Z"}]},
        {"name": "worker (2)", "steps": [
            {"name": "Node setup", "started_at": "2026-11-07T02:37:30Z", "completed_at": "2026-11-07T02:41:30Z"},
            {"name": "Queued", "started_at": None, "completed_at": None}]},
        {"name": "plan", "steps": [{"name": "Node setup", "started_at": "2026-11-07T02:30:00Z",
                                    "completed_at": "2026-11-07T03:30:00Z"}]}]}
    return tmp_path, steps


def test_compute_vs_network_maps_workers_by_address(run):
    root, steps = run
    md, d = tt.build(str(root), steps)
    w1 = d["workers"][0]
    assert (w1["worker"], w1["addr"], w1["guard_s"], w1["guard_n"]) == (1, "bore.pub:12227", 1.7, 95)
    assert (w1["connect_s"], w1["connect_n"], w1["network_s"]) == (6.8, 89, 5.1)
    assert d["workers"][1]["network_s"] == 10.0 and d["workers"][2]["network_s"] is None
    assert d["totals"] == {"guard_s": 4.1, "connect_s": 19.2, "network_s": 15.1}
    assert d["connect_all_s"] == 19.2  # worker 3 has no connect timing
    assert d["slowest_worker"] == {"worker": 2, "addr": "bore.pub:40001", "connect_s": 12.4,
                                   "guard_s": 2.4, "network_s": 10.0}
    assert "connect total, all workers (2)" in md and "workers with both guard and connect (2)" in md
    # forwards from DSpark: 8 written - 3 accepted + 1 prefill = 6
    assert (d["forwards"], d["forwards_source"].split()[0]) == (6, "DSpark")
    assert d["per_layer_ms"] == {"guard": round(4100 / 240, 1), "connect": 80.0,
                                 "network": round(15100 / 240, 1), "connect_all_serial": 80.0}
    assert d["t1_ms"] == 49000.0
    assert "not a per-layer measurement" in md and "ليس قياساً لكل طبقة" in md


def test_tokens_startup_and_stability(run):
    root, steps = run
    md, d = tt.build(str(root), steps)
    t = d["tokens"]
    assert (t["source"], t["first_s"], t["s_per_token"]) == ("tokens", 100.0, 51.2)
    assert (t["gap_p50"], t["gap_p90"]) == (50.0, 60.0)
    assert d["loaded_s"] == 255.35
    assert d["steps"]["coordinator"] == {"Node setup": 180.0, "Run the requested task": 630.0}
    assert d["steps"]["workers"]["Node setup"] == (120.0, 180.0, 240.0)   # plan job ignored
    s = d["stability"]
    assert s["early_closes"] == 1 and s["pipe_errors"] == 1 and s["timeouts"] == 1
    assert s["resource"] == 1 and s["never_connected"] == [3] and s["expected"] == 3
    assert "| Node setup | 180.0 | 120.0 / 180.0 / 240.0 |" in md


def test_fallbacks_without_tokens_steps_or_dspark(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"wall_s": 669.0, "completion_tokens": 8}))
    (tmp_path / "engine.err").write_text("PROF 669.0 20 8 0 0 1 2 3 9\n")
    md, d = tt.build(str(tmp_path))
    assert d["tokens"]["source"] == "wall_s/completion_tokens" and d["tokens"]["s_per_token"] == 83.6
    assert (d["forwards"], d["forwards_source"]) == (9, "PROF")
    assert "no job-steps JSON" in md and d["loaded_s"] is None


def test_cli_writes_files(run, tmp_path_factory):
    root, steps = run
    out = tmp_path_factory.mktemp("o")
    (out / "jobs.json").write_text(json.dumps(steps))
    assert tt.main([str(root), "--steps", str(out / "jobs.json"), "--out", str(out / "t.md"),
                    "--json", str(out / "t.json")]) == 0
    assert "## (ج)" in (out / "t.md").read_text()
    assert json.loads((out / "t.json").read_text())["workers"][0]["network_s"] == 5.1


A9 = __import__("pathlib").Path(__file__).parent / "fixtures" / "asa_a9"


@pytest.mark.skipif(not (A9 / "swarm-result" / "engine.err").exists(), reason="A9 fixture absent")
def test_real_a9_engine_result_and_steps():
    steps = json.loads((A9 / "jobs.json").read_text())
    md, d = tt.build(str(A9), steps)
    assert d["loaded_s"] == 219.58 and d["t1_ms"] == 49030.0
    assert (d["forwards"], d["t1_equivalent_forwards"]) == (3, 8.4)   # 7 written - 5 accepted + 1
    assert d["tokens"]["source"] == "wall_s/completion_tokens" and d["tokens"]["s_per_token"] == 78.8
    assert d["steps"]["coordinator"]["Run the requested task"] == 630.0
    staging = next(v for k, v in d["steps"]["workers"].items() if k.startswith("Node setup + stage"))
    assert staging == (136.0, 159.0, 182.0)


@pytest.mark.skipif(not (A9 / "worker-logs-1" / "guard.log").exists(), reason="A9 logs absent")
def test_real_a9_compute_vs_network():
    md, d = tt.build(str(A9), json.loads((A9 / "jobs.json").read_text()))
    w = {r["worker"]: r for r in d["workers"]}
    assert len(w) == 19 and all(r["route"] == "relay" for r in w.values())
    assert (w[1]["addr"], w[1]["guard_s"], w[1]["guard_n"], w[1]["connect_s"], w[1]["connect_n"],
            w[1]["network_s"]) == ("bore.pub:12227", 1.7, 95, 6.8, 89, 5.1)
    assert (w[2]["addr"], w[2]["guard_s"], w[2]["guard_n"], w[2]["connect_s"], w[2]["network_s"]) == \
        ("bore.pub:34071", 6.2, 70, 11.5, 5.3)
    assert d["connect_all_s"] == 214.3                     # the serial wait before the patch
    assert d["totals"] == {"guard_s": 7.9, "connect_s": 18.3, "network_s": 10.4}
    assert (d["slowest_worker"]["worker"], d["slowest_worker"]["connect_s"]) == (9, 17.7)
    assert d["per_layer_ms"]["connect_all_serial"] == round(214300 / 120, 1)
    assert d["stability"]["never_connected"] == [] and d["stability"]["expected"] == 19
    assert d["stability"]["timeouts"] == 0 and d["stability"]["resource"] == 0


def test_stage_b_token_stream_and_partial(tmp_path):
    toks = [{"t": 60.0, "kind": "content", "text": "a"}, {"t": 61.0, "kind": "meta", "text": ""},
            {"t": 100.0, "kind": "content", "text": "b"}, {"t": 130.0, "kind": "content", "text": "c"}]
    (tmp_path / "partial.json").write_text(json.dumps(
        {"tokens": toks, "stopped": True, "finish_reason": "length", "wall_s": 130.0}))
    (tmp_path / "result.json").write_text(json.dumps({"wall_s": None}))  # run cut short
    md, d = tt.build(str(tmp_path))
    t = d["tokens"]
    assert t["source"] == "partial.json tokens" and t["n"] == 3        # meta item dropped
    assert (t["first_s"], t["gap_p50"], t["s_per_token"]) == (60.0, 35.0, 43.3)
    assert (t["stopped"], t["finish_reason"]) == (True, "length")
    assert "| التوقف · stopped / finish_reason | True / length |" in md
    assert "no paired guard/connect logs" in md


def test_connect_total_includes_workers_without_guard_log(run):
    root, steps = run
    (root / "worker-logs-2" / "guard.log").unlink()     # like A9: only some guard logs kept
    md, d = tt.build(str(root), steps)
    assert d["connect_all_s"] == 19.2 and d["totals"]["connect_s"] == 6.8
    assert "workers with both guard and connect (1)" in md
    assert d["slowest_worker"]["worker"] == 2 and d["slowest_worker"]["guard_s"] is None
