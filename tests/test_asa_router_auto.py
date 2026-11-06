"""ASA stage 7: asa-auto routing heuristic and the generated LiteLLM config."""

import asyncio
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROUTER = ROOT / "asa" / "router"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"asa_{name}", ROUTER / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ar = _load("auto_route")


def _req(text, **extra):
    return {"model": "asa-auto", "messages": [{"role": "user", "content": text}], **extra}


@pytest.mark.parametrize("text", ["hi, how are you?", "مرحبا، كيف حالك؟", "what's the capital of France?"])
def test_simple_goes_fast(text):
    assert ar.explain(_req(text)) == ("asa-fast", "short simple prompt")


@pytest.mark.parametrize("text,reason", [
    ("prove that sqrt 2 is irrational", "hard task"),
    ("حلل هذا النص", "hard task"),
    ("اكتب كود يرتب قائمة", "hard task"),
    ("fix this:\n```py\nx=1\n```", "code"),
    ("what is $\\frac{1}{2}$?", "math"),
    ("x" * 401, "long prompt"),
])
def test_hard_goes_strong(text, reason):
    assert ar.explain(_req(text)) == ("asa-deepseek", reason)


def test_tools_images_long_chats_and_empty_go_strong():
    assert ar.choose(_req("hi", tools=[{"type": "function"}])) == "asa-deepseek"
    img = {"model": "asa-auto", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "x"}}]}]}
    assert ar.explain(img) == ("asa-deepseek", "non-text content")
    chat = {"messages": [{"role": "user", "content": "hi"}] * 7}
    assert ar.explain(chat) == ("asa-deepseek", "long conversation")
    assert ar.choose({"messages": []}) == "asa-deepseek"


def test_thresholds_from_env(monkeypatch):
    monkeypatch.setenv("ASA_AUTO_FAST_MAX_CHARS", "5")
    assert ar.explain(_req("hello there")) == ("asa-deepseek", "long prompt")


def test_hook_only_rewrites_asa_auto():
    hook = ar.proxy_handler_instance
    data = asyncio.run(hook.async_pre_call_hook(None, None, _req("hi"), "acompletion"))
    assert data["model"] == "asa-fast"
    assert data["metadata"]["asa_auto"] == {"model": "asa-fast", "reason": "short simple prompt"}
    other = {"model": "asa-deepseek", "messages": [{"role": "user", "content": "hi"}]}
    assert asyncio.run(hook.async_pre_call_hook(None, None, other, "acompletion"))["model"] == "asa-deepseek"


def test_build_config(monkeypatch, tmp_path):
    yaml = pytest.importorskip("yaml")
    bc = _load("build_config")
    src = yaml.safe_load((ROUTER / "litellm.yaml").read_text())
    out = bc.build(src)
    auto = next(m for m in out["model_list"] if m["model_name"] == "asa-auto")
    strong = next(m for m in out["model_list"] if m["model_name"] == "asa-deepseek")
    assert auto["litellm_params"] == strong["litellm_params"]
    assert out["litellm_settings"]["callbacks"] == ["auto_route.proxy_handler_instance"]
    assert out["litellm_settings"]["cache"] is True  # exact cache kept
    assert bc.build(out) == out  # idempotent
    assert "asa-auto" not in {m["model_name"] for m in src["model_list"]}  # source untouched


def test_build_config_cli(tmp_path):
    pytest.importorskip("yaml")
    run = lambda **env: subprocess.run(
        [sys.executable, str(ROUTER / "build_config.py"), str(tmp_path)],
        env={"PATH": "/usr/bin:/bin", **env}, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert run() == str(ROUTER / "litellm.yaml")
    assert run(ODYSSEUS_LOCAL_LLM="1") == str(tmp_path / "litellm.yaml")
    assert (tmp_path / "auto_route.py").read_text() == (ROUTER / "auto_route.py").read_text()
    assert run(ODYSSEUS_LOCAL_LLM="1", ASA_ROUTER_AUTO="0") == str(ROUTER / "litellm.yaml")
