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


def ask(base: str, key: str, model: str, prompt: str, max_tokens: int,
        deadline_s: float | None = None, partial: str | None = None,
        thinking: bool = False) -> dict:
    """Stream the completion, timestamping each delta, and stop at deadline_s.

    The partial file is rewritten after every delta, so a job that times out
    still leaves the tokens produced so far. Closing the connection at the
    deadline only ends the request early; the tokens already produced are
    unchanged (greedy decoding, temperature 0).
    """
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True}}
    if thinking:
        payload["enable_thinking"] = True
    req = urllib.request.Request(f"{base}/chat/completions", json.dumps(payload).encode(), {
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t = time.time()
    tokens, content, reasoning, usage, finish, stopped = [], [], [], {}, None, None

    def save():
        if partial:
            with open(partial, "w", encoding="utf-8") as f:
                json.dump({"elapsed_s": round(time.time() - t, 1), "reasoning": "".join(reasoning),
                           "answer": "".join(content), "tokens": tokens}, f, ensure_ascii=False)

    with urllib.request.urlopen(req, timeout=6 * 3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                finish = choice.get("finish_reason") or finish
                for field, kind, sink in (("reasoning_content", "reasoning", reasoning),
                                          ("content", "content", content)):
                    piece = delta.get(field)
                    if piece:
                        sink.append(piece)
                        tokens.append({"t": round(time.time() - t, 2), "kind": kind, "text": piece})
            save()
            if deadline_s is not None and time.time() - t > deadline_s:
                stopped = "deadline"
                break
    wall = time.time() - t
    n = usage.get("completion_tokens") or len(tokens)
    return {"answer": "".join(content), "reasoning": "".join(reasoning), "wall_s": round(wall, 1),
            "completion_tokens": n, "prompt_tokens": usage.get("prompt_tokens"),
            "tok_per_s_end_to_end": round(n / wall, 3) if wall else None,
            "finish_reason": finish, "stopped": stopped, "tokens": tokens}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--key", required=True)
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="result.json")
    ap.add_argument("--deadline-s", type=float, default=None,
                    help="stop reading after this many seconds of generation (partial result kept)")
    ap.add_argument("--partial", default=None, help="file rewritten with the tokens so far")
    ap.add_argument("--thinking", action="store_true", help="ask for the model's own reasoning trace")
    a = ap.parse_args(argv)
    ready = wait_ready(a.base, a.key)
    print(f"[ask] endpoint ready after {ready:.0f}s", file=sys.stderr, flush=True)
    res = ask(a.base, a.key, a.model, open(a.prompt_file, encoding="utf-8").read(), a.max_tokens,
              a.deadline_s, a.partial, a.thinking)
    res["ready_s"] = round(ready, 1)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(res["answer"])
    print(json.dumps({k: v for k, v in res.items() if k not in ("answer", "reasoning", "tokens")}, ensure_ascii=False), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
