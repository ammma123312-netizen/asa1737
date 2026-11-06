"""Benchmark any OpenAI-compatible server with the fixed ASA prompt set.

Used to compare Colibrì (`coli serve`) and llama.cpp (`llama-server`) on the
same machine and model. Greedy decoding (temperature 0) so runs with and
without a draft model must produce identical text.

    python -m asa.bench_api --port 8090
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

PROMPTS = [
    ("ar", "ما هي عاصمة المملكة العربية السعودية؟ أجب في جملتين.", 120),
    ("en", "Explain in 3 sentences why mixture-of-experts models are cheaper to run.", 120),
    ("code", "Write a bash one-liner that counts lines in all .py files, then explain it briefly.", 200),
    ("code2", "Write a Python function that checks whether a number is prime.", 200),
]


def run(base_url: str, model: str | None) -> list[dict]:
    rows = []
    for tag, prompt, max_tokens in PROMPTS:
        body = {"messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0}
        if model:
            body["model"] = model
        req = urllib.request.Request(f"{base_url}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t = time.time()
        with urllib.request.urlopen(req, timeout=1800) as r:
            resp = json.load(r)
        dt = time.time() - t
        n = resp["usage"]["completion_tokens"]
        row = {"tag": tag, "tokens": n, "e2e_tok_s": round(n / dt, 2),
               "decode_tok_s": round(resp.get("timings", {}).get("predicted_per_second", 0), 2),
               "text": resp["choices"][0]["message"]["content"]}
        print(f"{tag:5} {n:4d} tok  e2e {row['e2e_tok_s']:5.2f} tok/s  "
              f"decode {row['decode_tok_s']:5.2f} tok/s  | {row['text'][:60]!r}", flush=True)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    run(f"http://{a.host}:{a.port}", a.model)


if __name__ == "__main__":
    main()
