#!/usr/bin/env python3
"""Build the LiteLLM config the container actually runs (ASA stage 7).

Reads asa/router/litellm.yaml unchanged and, when auto routing is on, writes a copy to the run
directory with:
  * an ``asa-auto`` model (same backend as asa-deepseek, so it still answers if the hook is
    skipped), and
  * ``litellm_settings.callbacks: auto_route.proxy_handler_instance``.
auto_route.py is copied next to it because LiteLLM resolves callback modules relative to the
config file. Prints the path of the config to run.

Auto routing needs asa-fast (the bundled local LLM), so it defaults to ``ODYSSEUS_LOCAL_LLM``;
force it with ``ASA_ROUTER_AUTO=1`` or ``0``.

Usage: build_config.py [RUN_DIR]   (default $ASA_ROUTER_RUN_DIR or ~/.cache/asa-router)
"""

from __future__ import annotations

import copy
import os
import shutil
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "litellm.yaml")


def auto_enabled() -> bool:
    return os.getenv("ASA_ROUTER_AUTO", os.getenv("ODYSSEUS_LOCAL_LLM", "0")) == "1"


def build(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    models = cfg.setdefault("model_list", [])
    names = {m.get("model_name") for m in models}
    if "asa-auto" not in names:
        strong = next(m for m in models if m.get("model_name") == "asa-deepseek")
        models.append({"model_name": "asa-auto",
                       "litellm_params": copy.deepcopy(strong["litellm_params"])})
    settings = cfg.setdefault("litellm_settings", {}) or {}
    cfg["litellm_settings"] = settings
    callbacks = settings.get("callbacks") or []
    if isinstance(callbacks, str):
        callbacks = [callbacks]
    if "auto_route.proxy_handler_instance" not in callbacks:
        callbacks.append("auto_route.proxy_handler_instance")
    settings["callbacks"] = callbacks
    return cfg


def main(argv: list[str]) -> int:
    if not auto_enabled():
        print(SOURCE)
        return 0
    run_dir = (argv[0] if argv else None) or os.getenv("ASA_ROUTER_RUN_DIR") or \
        os.path.join(os.path.expanduser("~"), ".cache", "asa-router")
    os.makedirs(run_dir, exist_ok=True)
    with open(SOURCE, encoding="utf-8") as f:
        cfg = build(yaml.safe_load(f))
    out = os.path.join(run_dir, "litellm.yaml")
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    shutil.copyfile(os.path.join(HERE, "auto_route.py"), os.path.join(run_dir, "auto_route.py"))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
