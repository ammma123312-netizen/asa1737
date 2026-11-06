#!/usr/bin/env python3
"""Register the in-container ASA router (LiteLLM, asa/router/litellm.yaml) with Odysseus.

Stage 6 of docs/asa/README.md: behind ``ASA_ROUTER=1`` the container starts LiteLLM on
127.0.0.1:$ASA_ROUTER_PORT (default 4000) and this script makes it show up in the model
picker as a shared endpoint, so the user can pick ``asa-deepseek`` (or ``asa-fast``) without a
Settings round trip.

The proxy needs a master key (LiteLLM refuses to run unauthenticated). It is not a
subscription, just a random string. Order of precedence:
  1. ``LITELLM_MASTER_KEY`` from the environment;
  2. the key file ``$ASA_ROUTER_KEY_FILE`` (default ``data/asa_router.key``), created with mode
     0600 on first run so the key survives restarts and HF persistence snapshots.

``--print-key`` prints the key (creating the file if needed) and exits; serve.sh uses it so the
proxy and the registered endpoint always agree.

Idempotent. It never changes the user's default model: asa-deepseek streams a 763B model and
can take minutes per answer, so it is offered, not imposed.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import uuid

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

ENDPOINT_NAME = "ASA Router (LiteLLM)"
DEEPSEEK_ALIAS = "asa-deepseek"
FAST_ALIAS = "asa-fast"
AUTO_ALIAS = "asa-auto"


def enabled() -> bool:
    return os.getenv("ASA_ROUTER", "0") == "1"


def base_url() -> str:
    return f"http://127.0.0.1:{os.getenv('ASA_ROUTER_PORT', '4000')}/v1"


def key_file() -> str:
    return os.getenv("ASA_ROUTER_KEY_FILE") or os.path.join(DATA_DIR, "asa_router.key")


def ensure_key() -> str:
    env_key = os.getenv("LITELLM_MASTER_KEY", "").strip()
    if env_key:
        return env_key
    path = key_file()
    try:
        with open(path, "r", encoding="utf-8") as f:
            key = f.read().strip()
        if key:
            return key
    except FileNotFoundError:
        pass
    # LiteLLM expects master keys to start with "sk-".
    key = "sk-asa-" + secrets.token_hex(16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(key + "\n")
    return key


def models() -> list[str]:
    names = [DEEPSEEK_ALIAS]
    # asa-fast points at the bundled llama-cpp server; only offer it when that is running.
    if os.getenv("ODYSSEUS_LOCAL_LLM", "0") == "1":
        names.append(FAST_ALIAS)
    # Same default as build_config.auto_enabled(): auto routing needs asa-fast.
    if os.getenv("ASA_ROUTER_AUTO", os.getenv("ODYSSEUS_LOCAL_LLM", "0")) == "1":
        names.append(AUTO_ALIAS)
    return names


def register(key: str) -> str:
    """Create or refresh the endpoint row. Returns a short status word."""
    sys.path.insert(0, BASE_DIR)
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{os.path.join(DATA_DIR, 'app.db')}")
    from core.database import Base, ModelEndpoint, SessionLocal, engine

    Base.metadata.create_all(bind=engine)
    url = base_url()
    wanted = models()
    names = json.dumps(wanted)
    db = SessionLocal()
    try:
        row = db.query(ModelEndpoint).filter(ModelEndpoint.base_url == url).one_or_none()
        if row is None:
            db.add(ModelEndpoint(
                id=str(uuid.uuid4()),
                name=ENDPOINT_NAME,
                base_url=url,
                api_key=key,
                is_enabled=True,
                pinned_models=names,
                cached_models=names,
                model_type="llm",
                endpoint_kind="local",
                model_refresh_mode="auto",
                # Colibri's OpenAI server does not do tool calls.
                supports_tools=False,
                owner=None,
            ))
            db.commit()
            return "created"
        status = "unchanged"
        if row.api_key != key:
            # Key rotated via LITELLM_MASTER_KEY: keep the endpoint working.
            row.api_key = key
            status = "key-updated"
        pinned = json.loads(row.pinned_models or "[]")
        missing = [m for m in wanted if m not in pinned]
        if missing:
            # e.g. asa-auto turned on later; add without dropping the admin's own pins.
            row.pinned_models = json.dumps(pinned + missing)
            status = "models-updated" if status == "unchanged" else status
        if status != "unchanged":
            db.commit()
        return status
    finally:
        db.close()


def main(argv: list[str]) -> int:
    if not enabled():
        return 0
    key = ensure_key()
    if "--print-key" in argv:
        print(key)
        return 0
    try:
        status = register(key)
    except Exception as exc:  # never block container start
        print(f"[asa-router] could not register endpoint: {exc}", file=sys.stderr)
        return 0
    print(f"[asa-router] endpoint '{ENDPOINT_NAME}' at {base_url()}: {status} "
          f"(models={', '.join(models())})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
