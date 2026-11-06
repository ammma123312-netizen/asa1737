"""Petals client for Google Colab (UNTESTED here — see notes).

Petals (https://github.com/bigscience-workshop/petals) splits a huge model's
layers across volunteer GPU servers. Your machine only runs the embeddings and
sends activations through the swarm, so a free Colab can talk to a 405B model.

Why untested: Petals peers talk libp2p over raw TCP, and the container this was
written in only allows HTTPS, so neither the swarm nor health.petals.dev could
be reached. Run it in Colab:

    !pip install -q git+https://github.com/bigscience-workshop/petals
    !wget -q https://raw.githubusercontent.com/<you>/asa/main/asa/petals/colab_client.py
    !python colab_client.py --list
    !python colab_client.py --model <name from --list> --prompt "..."

Reality check before relying on it:
  * The public swarm is volunteer-run; which models are served (if any)
    changes over time. `--list` asks the health monitor first.
  * Speed depends on the slowest server in your chain (typically a few
    tokens/s at best for 70B+, often less).
  * Your prompts travel through strangers' servers. Do not send private data;
    for that, run a private swarm (`python -m petals.cli.run_server` on your
    own machines with `--initial_peers` pointing at each other).
  * Llama models are gated on Hugging Face: run `huggingface-cli login` with a
    token that has accepted the model licence.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

HEALTH_URL = "https://health.petals.dev/api/v1/state"


def list_models() -> list[str]:
    """Return model names the public health monitor reports as served."""
    with urllib.request.urlopen(HEALTH_URL, timeout=30) as r:
        state = json.load(r)
    names = []
    for rep in state.get("model_reports", []):
        name = rep.get("name") or rep.get("repository") or rep.get("dht_prefix")
        status = rep.get("state", "?")
        print(f"{status:10} {name}")
        if name and status in ("healthy", "online"):
            names.append(name)
    if not state.get("model_reports"):
        print(json.dumps(state, indent=1)[:2000])
    return names


def generate(model_name: str, prompt: str, max_new_tokens: int) -> str:
    from petals import AutoDistributedModelForCausalLM  # heavy import, Colab only
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    out = model.generate(ids, max_new_tokens=max_new_tokens)
    return tok.decode(out[0])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Petals swarm client")
    ap.add_argument("--list", action="store_true", help="show models the public swarm serves")
    ap.add_argument("--model", help="model repo id served by the swarm")
    ap.add_argument("--prompt", default="The capital of Saudi Arabia is")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    a = ap.parse_args(argv)
    if a.list or not a.model:
        names = list_models()
        if not names:
            print("No healthy public models reported. The swarm may be down; "
                  "consider a private swarm.", file=sys.stderr)
        return 0
    print(generate(a.model, a.prompt, a.max_new_tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main())
