"""Timing table for one ASA swarm run, from its downloaded artifacts (Arabic + English Markdown).

    python -m asa.swarm.timing_table RUN_DIR [--steps jobs.json] [--layers 40] [--out table.md]

RUN_DIR holds the run's artifacts unzipped, each in its own folder (as `gh run download` lays
them out); file names are matched anywhere below it:
    swarm-result/   result.json, engine.err, rendezvous.log, connect.log
    worker-logs-K/  guard.log, worker.log, bore.log
--steps is the GitHub API's jobs listing ({"jobs": [...]} or a plain list), with step names and
started_at/completed_at.

Sections:
  (a) startup and staging: per-step durations, coordinator and workers (min/median/max), and
      the engine's "loaded in Xs";
  (b) seconds per token: from result.json "tokens" ([{t, text}], t = seconds since the request);
      falls back to wall_s / completion_tokens;
  (c) compute vs network: per worker, the guard's reply time (worker side) against the
      coordinator's `connect` time for that worker's address (mapped through rendezvous.log);
      network = connect - guard. Then an average per layer, totals / (forwards x layers):
      an average, not a per-layer measurement;
  (d) stability: timeouts, pipes closed early, OOM / EDQUOT / ENOSPC / Killed, workers that
      never connected.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from datetime import datetime

TIMING = re.compile(r"\[timing\] (guard|connect) (.+?) (\{.*\})\s*$")
RDV_WORKER = re.compile(r"\[rendezvous\] worker (\d+): (\S+) (\S+)")
RDV_COUNT = re.compile(r"\[rendezvous\] (\d+)/(\d+) workers")
CONNECTED = re.compile(r"\[connect\] authenticated to (\S+)")
REFUSED = re.compile(r"\[connect\] (\S+) refused us")
PIPE_CLOSED = re.compile(r"\[pipe\] (connect|guard) (.+?) (->|<-) closed by peer")
PIPE_ERROR = re.compile(r"\[pipe\] (connect|guard) (.+?) (->|<-) error after")
ENGINE_WALL = re.compile(r"expert I/O:.*\bwall ([0-9.]+)s")
LOADED = re.compile(r"loaded in ([0-9.]+)s\b")
T1 = re.compile(r"DSpark gate:\s*T\(1\)\s*([0-9.]+)\s*ms")
DSPARK = re.compile(r"DSpark: (\d+) of (\d+) drafts accepted this turn \((\d+) written\)")
PROF = re.compile(r"^PROF (?:\S+ ){8}(\d+)\s*$", re.M)
TIMEOUT = re.compile(r"time(?:d)? ?out|TimeoutError", re.I)
RESOURCE = re.compile(r"Out of memory|\bOOM\b|oom-kill|EDQUOT|Disk quota exceeded|ENOSPC|"
                      r"No space left on device|Cannot allocate memory|\bKilled\b")


# ------------------------------------------------------------------ reading
def find_files(root: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            out.setdefault(f, []).append(os.path.join(dp, f))
    for v in out.values():
        v.sort()
    return out


def read(path: str | None) -> str:
    if not path:
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def worker_index(path: str) -> int | None:
    m = re.search(r"worker-logs-(\d+)", path)
    return int(m.group(1)) if m else None


def last_timings(text: str, kind: str) -> dict[str, dict]:
    """label -> the last [timing] summary for that label."""
    out = {}
    for line in text.splitlines():
        m = TIMING.search(line)
        if m and m.group(1) == kind:
            try:
                out[m.group(2)] = json.loads(m.group(3))
            except json.JSONDecodeError:
                pass
    return out


def main_timing(text: str, kind: str) -> dict | None:
    """The connection that carried the run: of each label's last report, the one with most calls."""
    t = [v for v in last_timings(text, kind).values() if v.get("n")]
    return max(t, key=lambda v: v["n"]) if t else None


def rendezvous_map(text: str) -> dict[int, tuple[str, str]]:
    """worker K -> (route kind, address)."""
    return {int(k): (how, addr) for k, how, addr in RDV_WORKER.findall(text)}


def _parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def step_durations(jobs) -> tuple[dict[str, float], dict[str, list[float]]]:
    """(coordinator step -> seconds, worker step -> [seconds per worker])."""
    if isinstance(jobs, dict):
        jobs = jobs.get("jobs", [])
    coord, workers = {}, {}
    for j in jobs:
        name = str(j.get("name", ""))
        for s in j.get("steps") or []:
            a, b = _parse_ts(s.get("started_at")), _parse_ts(s.get("completed_at"))
            if not (a and b):
                continue
            d = (b - a).total_seconds()
            if name == "coordinator":
                coord[s["name"]] = d
            elif name.startswith("worker"):
                workers.setdefault(s["name"], []).append(d)
    return coord, workers


# ------------------------------------------------------------------ metrics
NON_TEXT_KINDS = {"meta", "stop", "done", "error", "usage", "role"}


def token_stats(result: dict) -> dict:
    """result.json (or partial.json) of ask.py: stage B adds tokens [{t, kind, text}],
    "stopped" and "finish_reason"; older runs only have wall_s and completion_tokens."""
    toks = [x for x in result.get("tokens") or []
            if isinstance(x, dict) and "t" in x and x.get("kind") not in NON_TEXT_KINDS]
    out = {"source": None, "first_s": None, "gap_p50": None, "gap_p90": None, "s_per_token": None,
           "n": len(toks), "stopped": result.get("stopped"),
           "finish_reason": result.get("finish_reason")}
    if len(toks) >= 1:
        ts = [float(x["t"]) for x in toks]
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        out.update(source="tokens", first_s=round(ts[0], 1), s_per_token=round(ts[-1] / len(ts), 1))
        if gaps:
            gs = sorted(gaps)
            out["gap_p50"] = round(statistics.median(gs), 1)
            out["gap_p90"] = round(gs[min(len(gs) - 1, int(0.9 * len(gs)))], 1)
        return out
    wall, n = result.get("wall_s"), result.get("completion_tokens")
    if wall and n:
        out.update(source="wall_s/completion_tokens", s_per_token=round(wall / n, 1), n=n)
    return out


def forward_count(engine: str, result: dict) -> tuple[int | None, str]:
    prof = PROF.findall(engine)
    if prof:
        return sum(int(x) for x in prof), "PROF"
    d = DSPARK.findall(engine)
    if d:
        accepted, _, written = map(int, d[-1])
        return written - accepted + 1, "DSpark (written - accepted + 1 prefill)"
    n = result.get("completion_tokens")
    return (n + 1, "completion_tokens + 1 prefill") if n else (None, "unknown")


def stability(files: dict[str, list[str]], expected: int | None, rdv: dict, connect_text: str,
              guard_by_worker: dict[int, str]) -> dict:
    logs = [p for name, ps in files.items() for p in ps if name.endswith((".log", ".err"))]
    timeouts, resource, errors = [], [], []
    for p in logs:
        for line in read(p).splitlines():
            if TIMEOUT.search(line):
                timeouts.append((os.path.relpath(p), line.strip()))
            if RESOURCE.search(line):
                resource.append((os.path.relpath(p), line.strip()))
            if PIPE_ERROR.search(line):
                errors.append((os.path.relpath(p), line.strip()))
    # Colibri keeps one connection per worker for the whole run, so every authentication to
    # an address after its first means the previous connection dropped mid-run.
    auths = CONNECTED.findall(connect_text)
    authed = set(auths)
    early = sum(auths.count(a) - 1 for a in authed)
    never = []
    total = expected or (max(rdv) if rdv else max(guard_by_worker, default=0))
    for k in range(1, total + 1):
        addr = rdv.get(k, (None, None))[1]
        accepted = "[guard] accepted" in guard_by_worker.get(k, "")
        if not addr or addr not in authed or (k in guard_by_worker and not accepted):
            never.append(k)
    return {"timeouts": timeouts, "pipe_errors": errors, "early_closes": early,
            "resource": resource, "never_connected": never, "expected": total}


# ------------------------------------------------------------------ table
def _mmm(xs):
    xs = sorted(xs)
    return (round(xs[0], 1), round(statistics.median(xs), 1), round(xs[-1], 1)) if xs else None


def _f(v, nd=1):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def build(run_dir: str, steps=None, layers: int = 40) -> tuple[str, dict]:
    files = find_files(run_dir)
    pick = lambda n: (files.get(n) or [None])[0]  # noqa: E731
    def load_json(name):
        try:
            return json.loads(read(pick(name)) or "{}")
        except json.JSONDecodeError:
            return {}

    result, partial = load_json("result.json"), load_json("partial.json")
    # A run that was stopped or failed may only have the partial stream.
    timing_src = result if result.get("tokens") or not partial.get("tokens") else partial
    engine, connect_text = read(pick("engine.err")), read(pick("connect.log"))
    rdv_text = read(pick("rendezvous.log"))
    rdv = rendezvous_map(rdv_text)
    counts = RDV_COUNT.findall(rdv_text)
    expected = int(counts[-1][1]) if counts else None
    guard_by_worker = {worker_index(p): read(p) for p in files.get("guard.log", []) if worker_index(p)}
    conn = last_timings(connect_text, "connect")
    data: dict = {"result": {k: result.get(k) for k in ("wall_s", "completion_tokens")}}
    L: list[str] = []

    # (a) startup
    L += ["## (أ) الإقلاع والتجهيز · Startup and staging", ""]
    loaded = LOADED.findall(engine)
    data["loaded_s"] = float(loaded[0]) if loaded else None
    L.append(f"- تحميل النموذج عند المنسّق · engine `loaded in`: **{_f(data['loaded_s'], 2)} s**")
    if steps is not None:
        coord, work = step_durations(steps)
        data["steps"] = {"coordinator": coord, "workers": {k: _mmm(v) for k, v in work.items()}}
        L += ["", "| الخطوة · step | المنسّق · coordinator (s) | العمال · workers min / median / max (s) |",
              "|---|---|---|"]
        for name in list(coord) + [n for n in work if n not in coord]:
            mmm = _mmm(work.get(name, []))
            L.append(f"| {name} | {_f(coord.get(name))} | "
                     f"{' / '.join(map(str, mmm)) if mmm else '–'} |")
    else:
        L.append("- (no job-steps JSON given: step durations omitted)")
    L.append("")

    # (b) seconds per token
    ts = token_stats(timing_src)
    if timing_src is partial:
        ts["source"] = f"partial.json {ts['source']}"
    data["tokens"] = ts
    L += ["## (ب) الثواني لكل رمز · Seconds per token", "",
          "| المقياس · metric | القيمة · value |", "|---|---|",
          f"| أول رمز · first-token latency (s) | {_f(ts['first_s'])} |",
          f"| الفاصل الوسيط بين الرموز · median inter-token gap (s) | {_f(ts['gap_p50'])} |",
          f"| الفاصل p90 · p90 inter-token gap (s) | {_f(ts['gap_p90'])} |",
          f"| المتوسط الكلي · overall s/token | {_f(ts['s_per_token'])} |",
          f"| المصدر · source | {ts['source'] or '–'} ({ts['n']} tokens) |"]
    if ts["stopped"] is not None or ts["finish_reason"]:
        L.append(f"| التوقف · stopped / finish_reason | {ts['stopped']} / {ts['finish_reason'] or '–'} |")
    L.append("")

    # (c) compute vs network
    rows, tot_g, tot_c = [], 0.0, 0.0
    for k in sorted(set(guard_by_worker) | set(rdv)):
        g = main_timing(guard_by_worker.get(k, ""), "guard")
        how, addr = rdv.get(k, ("?", None))
        c = conn.get(addr) if addr else None
        net = round(c["total_s"] - g["total_s"], 1) if (g and c) else None
        rows.append({"worker": k, "route": how, "addr": addr,
                     "guard_s": g and g["total_s"], "guard_n": g and g["n"],
                     "connect_s": c and c.get("total_s"), "connect_n": c and c.get("n"),
                     "network_s": net})
        if g and c:
            tot_g += g["total_s"]
            tot_c += c["total_s"]
    fw, fw_src = forward_count(engine, result)
    t1 = T1.findall(engine)
    data.update(workers=rows, forwards=fw, forwards_source=fw_src,
                t1_ms=float(t1[-1]) if t1 else None,
                totals={"guard_s": round(tot_g, 1), "connect_s": round(tot_c, 1),
                        "network_s": round(tot_c - tot_g, 1)})
    L += ["## (ج) الحساب مقابل الشبكة · Compute vs network", "",
          "| العامل · worker | المسار · route | العنوان · address | حساب العامل · guard (s / calls) "
          "| انتظار المنسّق · connect (s / calls) | الشبكة · network (s) |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['worker']} | {r['route']} | {r['addr'] or '–'} | "
                 f"{_f(r['guard_s'])} / {_f(r['guard_n'])} | {_f(r['connect_s'])} / {_f(r['connect_n'])} | "
                 f"{_f(r['network_s'])} |")
    T = data["totals"]
    paired = sum(1 for r in rows if r["network_s"] is not None)
    # Every worker's connect time, guard.log or not: before the parallel-batch patch the
    # coordinator waited on workers one after another, so this sum is the serial wait.
    conn_all = [r for r in rows if r["connect_s"] is not None]
    data["connect_all_s"] = round(sum(r["connect_s"] for r in conn_all), 1)
    L.append(f"| **مجموع الانتظار لكل العمال · connect total, all workers ({len(conn_all)})** | | | "
             f"| **{_f(data['connect_all_s'])}** | |")
    L.append(f"| **المجموع للعمال ذوي القياسين · total, workers with both guard and connect "
             f"({paired})** | | | **{_f(T['guard_s'])}** | **{_f(T['connect_s'])}** | "
             f"**{_f(T['network_s'])}** |")
    if conn_all:
        slow = max(conn_all, key=lambda r: r["connect_s"])
        data["slowest_worker"] = {k: slow[k] for k in ("worker", "addr", "connect_s", "guard_s", "network_s")}
        L.append(f"\n- أبطأ عامل · slowest worker: **{slow['worker']}** ({slow['addr']}), connect "
                 f"**{_f(slow['connect_s'])} s**, guard {_f(slow['guard_s'])} s, network "
                 f"{_f(slow['network_s'])} s. When the coordinator overlaps its requests to all workers "
                 f"(parallel-batch patch), each layer waits roughly for the slowest worker, not the sum.")
    have_pairs = any(r["network_s"] is not None for r in rows)
    if not have_pairs:
        L.append("\n- لا توجد سجلات guard/connect مقترنة · no paired guard/connect logs found: "
                 "compute vs network not available")
    if fw and have_pairs:
        per = lambda s: round(1000 * s / (fw * layers), 1)  # noqa: E731
        data["per_layer_ms"] = {"guard": per(tot_g), "connect": per(tot_c), "network": per(tot_c - tot_g)}
        P = data["per_layer_ms"]
        L += ["", f"متوسط لكل طبقة · per-layer average = totals / ({fw} forwards × {layers} layers), "
                  f"forwards from {fw_src}:",
              f"- الحساب · compute **{P['guard']} ms**, الانتظار · wait **{P['connect']} ms**, "
              f"الشبكة · network **{P['network']} ms**"]
    if data["t1_ms"]:
        L.append(f"- T(1) من سطر DSpark · from the DSpark gate line: **{data['t1_ms'] / 1000:.1f} s** "
                 f"for one single-row forward")
        ew = ENGINE_WALL.findall(engine)
        wall, wsrc = (float(ew[-1]), "engine wall") if ew else ((timing_src or {}).get("wall_s"), "wall_s")
        if wall:
            data["t1_equivalent_forwards"] = round(wall / (data["t1_ms"] / 1000), 1)
            L.append(f"- مكافئ التمريرات · {wsrc} {wall:.1f} s / T(1) = **{data['t1_equivalent_forwards']}** "
                     f"single-row passes (a batched verify pass costs more than T(1), so this can "
                     f"exceed the forward count)")
    L += ["", "> هذا متوسط محسوب من المجاميع، وليس قياساً لكل طبقة. "
              "This is an average derived from totals, not a per-layer measurement.", ""]

    # (d) stability
    st = stability(files, expected, rdv, connect_text, guard_by_worker)
    data["stability"] = {k: (len(v) if isinstance(v, list) and k != "never_connected" else v)
                         for k, v in st.items()}
    L += ["## (د) الاستقرار · Stability", "",
          "| البند · item | العدد · count |", "|---|---|",
          f"| مهلات · timeouts | {len(st['timeouts'])} |",
          f"| أنابيب أُغلقت مبكراً ثم أعيد الاتصال · pipes closed then reconnected | {st['early_closes']} |",
          f"| أخطاء أنابيب · pipe errors | {len(st['pipe_errors'])} |",
          f"| ذاكرة/قرص · OOM / EDQUOT / ENOSPC / Killed | {len(st['resource'])} |",
          f"| عمال لم يتصلوا · workers never connected | {len(st['never_connected'])} of {st['expected']}"
          f"{' (' + ', '.join(map(str, st['never_connected'])) + ')' if st['never_connected'] else ''} |"]
    for title, items in (("timeouts", st["timeouts"]), ("pipe errors", st["pipe_errors"]),
                         ("OOM/disk", st["resource"])):
        for path, line in items[:5]:
            L.append(f"\n- {title}: `{path}`: `{line[:160]}`")
    return "\n".join(L) + "\n", data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Timing table for one ASA swarm run.")
    ap.add_argument("run_dir")
    ap.add_argument("--steps", help="GitHub API jobs JSON with step timestamps")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--out", help="write the Markdown here as well")
    ap.add_argument("--json", help="write the numbers here")
    a = ap.parse_args(argv)
    steps = json.load(open(a.steps, encoding="utf-8")) if a.steps else None
    md, data = build(a.run_dir, steps, a.layers)
    print(md)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(md)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
