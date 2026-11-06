"""List which hosted providers currently serve the ASA models, and at what price.

Free tiers and model ids change often; run this before relying on the router.

    python -m asa.check_providers
"""
from __future__ import annotations

import json
import urllib.request

WANT = ("deepseek-v4.1-flash", "deepseek-v4-pro", "glm-5.2", "glm-5.3", "kimi-k2")


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "asa-check-providers"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def openrouter() -> list[str]:
    out = []
    for m in _get("https://openrouter.ai/api/v1/models")["data"]:
        if any(w in m["id"] for w in WANT):
            p = m["pricing"]
            pin, pout = float(p["prompt"]) * 1e6, float(p["completion"]) * 1e6
            tag = "FREE" if pin == 0 and pout == 0 else f"${pin:.2f} in / ${pout:.2f} out per M"
            out.append(f"openrouter  {m['id']:40} {tag}")
    return out


def nvidia_nim() -> list[str]:
    return [f"nvidia-nim  {m['id']:40} free trial credits"
            for m in _get("https://integrate.api.nvidia.com/v1/models")["data"]
            if any(w in m["id"].lower() for w in WANT)]


def main() -> None:
    for name, fn in (("OpenRouter", openrouter), ("NVIDIA NIM", nvidia_nim)):
        try:
            rows = fn()
            print("\n".join(rows) if rows else f"{name}: no ASA models listed")
        except Exception as e:  # network/provider errors should not hide the others
            print(f"{name}: unavailable ({e})")


if __name__ == "__main__":
    main()
