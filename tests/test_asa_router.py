"""ASA stage 6: the LiteLLM router wiring (asa/router/seed_endpoint.py, serve.sh, entrypoint)."""

import importlib.util
import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "asa" / "router" / "seed_endpoint.py"
SERVE = ROOT / "asa" / "router" / "serve.sh"


def _load():
    spec = importlib.util.spec_from_file_location("asa_seed_endpoint", SEED)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def seed(monkeypatch, tmp_path):
    for var in ("ASA_ROUTER", "ASA_ROUTER_AUTO", "ASA_ROUTER_PORT", "LITELLM_MASTER_KEY", "ODYSSEUS_LOCAL_LLM"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ASA_ROUTER_KEY_FILE", str(tmp_path / "asa_router.key"))
    return _load()


def test_disabled_is_noop(seed, tmp_path, capsys):
    assert seed.main(["--print-key"]) == 0
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "asa_router.key").exists()


def test_key_file_created_private_and_reused(seed, tmp_path):
    key = seed.ensure_key()
    path = tmp_path / "asa_router.key"
    assert key.startswith("sk-") and len(key) > 20
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert seed.ensure_key() == key


def test_env_key_wins(seed, monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-from-env")
    assert seed.ensure_key() == "sk-from-env"
    assert not (tmp_path / "asa_router.key").exists()


def test_models_and_url(seed, monkeypatch):
    assert seed.models() == ["asa-deepseek"]
    monkeypatch.setenv("ODYSSEUS_LOCAL_LLM", "1")
    assert seed.models() == ["asa-deepseek", "asa-fast", "asa-auto"]
    monkeypatch.setenv("ASA_ROUTER_AUTO", "0")
    assert seed.models() == ["asa-deepseek", "asa-fast"]
    monkeypatch.setenv("ASA_ROUTER_PORT", "4123")
    assert seed.base_url() == "http://127.0.0.1:4123/v1"


def test_models_match_router_config():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((ROOT / "asa" / "router" / "litellm.yaml").read_text())
    names = {m["model_name"] for m in cfg["model_list"]}
    mod = _load()
    assert {mod.DEEPSEEK_ALIAS, mod.FAST_ALIAS} <= names
    assert cfg["general_settings"]["master_key"] == "os.environ/LITELLM_MASTER_KEY"


def test_register_creates_then_updates_key(tmp_path):
    pytest.importorskip("sqlalchemy")
    db = tmp_path / "app.db"
    script = textwrap.dedent(f"""
        import importlib.util, json, sys
        spec = importlib.util.spec_from_file_location("s", {str(SEED)!r})
        s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
        out = [s.register("sk-one"), s.register("sk-one"), s.register("sk-two")]
        from core.database import SessionLocal, ModelEndpoint
        db = SessionLocal()
        rows = db.query(ModelEndpoint).all()
        out.append([(r.name, r.base_url, r.api_key, json.loads(r.pinned_models), r.supports_tools)
                    for r in rows])
        print(json.dumps(out))
    """)
    env = dict(os.environ, DATABASE_URL=f"sqlite:///{db}", ASA_ROUTER="1",
               ODYSSEUS_LOCAL_LLM="1", PYTHONPATH=str(ROOT))
    env.pop("ASA_ROUTER_PORT", None)
    env.pop("ASA_ROUTER_AUTO", None)
    res = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr[-2000:]
    created, unchanged, updated, rows = json.loads(res.stdout.strip().splitlines()[-1])
    assert (created, unchanged, updated) == ("created", "unchanged", "key-updated")
    assert rows == [["ASA Router (LiteLLM)", "http://127.0.0.1:4000/v1", "sk-two",
                     ["asa-deepseek", "asa-fast", "asa-auto"], False]]


def test_serve_disabled_exits_cleanly():
    env = {k: v for k, v in os.environ.items() if k != "ASA_ROUTER"}
    res = subprocess.run(["sh", str(SERVE)], env=env, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0 and res.stdout == ""


def test_serve_execs_litellm_with_shared_key(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "litellm"
    fake.write_text('#!/bin/sh\necho "ARGS $*"\necho "KEY $LITELLM_MASTER_KEY"\n'
                    'echo "BASE $ASA_DEEPSEEK_BASE"\n')
    fake.chmod(0o755)
    key_file = tmp_path / "k"
    env = dict(os.environ, ASA_ROUTER="1", ASA_ROUTER_PORT="4555",
               ASA_ROUTER_KEY_FILE=str(key_file), ODYSSEUS_LOCAL_LLM="0", PATH=f"{fake_bin}:{os.environ['PATH']}")
    env.pop("LITELLM_MASTER_KEY", None)
    env.pop("ASA_DEEPSEEK_BASE", None)
    res = subprocess.run(["sh", str(SERVE)], env=env, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    lines = dict(line.split(" ", 1) for line in res.stdout.splitlines() if " " in line)
    assert "--host 127.0.0.1 --port 4555" in lines["ARGS"]
    # ODYSSEUS_LOCAL_LLM unset -> no auto routing -> the source config is used as is.
    assert lines["ARGS"].startswith(f"--config {SERVE.parent}/litellm.yaml")
    assert lines["KEY"] == key_file.read_text().strip()
    assert lines["BASE"] == "http://127.0.0.1:8000/v1"


def test_entrypoint_wires_router_before_exec():
    text = (ROOT / "docker" / "entrypoint.sh").read_text()
    seed_at = text.index("/app/asa/router/seed_endpoint.py")
    serve_at = text.index("/app/asa/router/serve.sh &")
    assert text.index('if [ "${ASA_ROUTER:-0}" = "1" ]') < seed_at < serve_at
    assert serve_at < text.index('exec gosu "$PUID:$PGID" "$@"')
