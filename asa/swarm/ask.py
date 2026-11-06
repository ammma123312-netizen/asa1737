"""Ask the swarm's OpenAI-compatible endpoint one question and record timing.

    python -m asa.swarm.ask --base http://127.0.0.1:8000/v1 --key "$ASA_API_KEY" \\
        --prompt-file prompt.txt --max-tokens 400 --out result.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def wait_ready(base: str, key: str, timeout: float = 3600) -> float:
    t, deadline = time.time(), time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
            urllib.request.urlopen(req, timeout=10).read()
            return time.time() - t
        except Exception:
            time.sleep(5)
    raise TimeoutError("endpoint never became ready")


def ask(base: str, key: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", body, {
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=6 * 3600) as r:
        resp = json.load(r)
    wall = time.time() - t
    usage = resp.get("usage", {})
    n = usage.get("completion_tokens", 0)
    return {"answer": resp["choices"][0]["message"]["content"], "wall_s": round(wall, 1),
            "completion_tokens": n, "prompt_tokens": usage.get("prompt_tokens"),
            "tok_per_s_end_to_end": round(n / wall, 3) if wall else None,
            "timings": resp.get("timings")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--key", required=True)
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="result.json")
    a = ap.parse_args(argv)
    ready = wait_ready(a.base, a.key)
    print(f"[ask] endpoint ready after {ready:.0f}s", file=sys.stderr, flush=True)
    res = ask(a.base, a.key, a.model, open(a.prompt_file, encoding="utf-8").read(), a.max_tokens)
    res["ready_s"] = round(ready, 1)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(res["answer"])
    print(json.dumps({k: v for k, v in res.items() if k != "answer"}, ensure_ascii=False), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
