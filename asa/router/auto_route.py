"""ASA stage 7: ``asa-auto`` picks asa-fast or asa-deepseek per request (LiteLLM pre-call hook).

asa-deepseek is native quality but slow (minutes per long answer); asa-fast answers instantly at
lower quality. ``asa-auto`` sends short, simple chat turns to asa-fast and everything that looks
like real work (long prompts, code, reasoning or analysis requests, long conversations, tool use,
images) to asa-deepseek. The rule is deliberately conservative: when in doubt it chooses quality.

This is a dependency-free heuristic in the spirit of RouteLLM; it needs no embedding model or
training data, so it costs nothing and its decisions are explainable (see ``explain``).

Tuning (environment):
  ASA_AUTO_FAST_MAX_CHARS   longest prompt (characters) still allowed on asa-fast (default 400)
  ASA_AUTO_FAST_MAX_TURNS   most messages in the conversation for asa-fast (default 6)
  ASA_AUTO_DEFAULT          model when the request carries no text (default asa-deepseek)

Loaded by build_config.py as ``litellm_settings.callbacks: auto_route.proxy_handler_instance``.
"""

from __future__ import annotations

import os
import re

AUTO = "asa-auto"
FAST = "asa-fast"
STRONG = "asa-deepseek"

# Words that signal a task needing the strong model (English and Arabic).
_HARD = re.compile(
    r"\b(prove|proof|derive|analy[sz]e|explain why|step by step|implement|refactor|debug|"
    r"algorithm|optimi[sz]e|translate|summari[sz]e|compare|design|architecture|calculate|"
    r"write (a|an|the)? ?(code|function|script|program|essay|report|article))\b"
    r"|برهن|اثبت|أثبت|حلل|حلّل|تحليل|اشرح لماذا|خطوة بخطوة|اكتب (كود|برنامج|دالة|مقال|تقرير)"
    r"|برمج|صحح الخطأ|ترجم|لخص|لخّص|قارن|صمم|احسب",
    re.IGNORECASE,
)
_CODE = re.compile(r"```|^\s*(def |class |import |function |#include|SELECT |<\?php)", re.M)
_MATH = re.compile(r"\\(frac|int|sum)|∫|∑|\$[^$]+\$")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _text(content) -> tuple[str, bool]:
    """Flatten OpenAI message content; second value is True if it holds non-text parts."""
    if isinstance(content, str):
        return content, False
    if isinstance(content, list):
        parts, other = [], False
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            else:
                other = True
        return "\n".join(parts), other
    return "", content is not None


def explain(data: dict) -> tuple[str, str]:
    """Return (model, reason) for an OpenAI-style request body."""
    messages = data.get("messages") or []
    if data.get("tools") or data.get("functions"):
        return STRONG, "tool use"
    if len(messages) > _env_int("ASA_AUTO_FAST_MAX_TURNS", 6):
        return STRONG, "long conversation"
    last = next((m for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"),
                None)
    if last is None:
        return os.getenv("ASA_AUTO_DEFAULT", STRONG), "no user message"
    text, non_text = _text(last.get("content"))
    if non_text:
        return STRONG, "non-text content"
    if not text.strip():
        return os.getenv("ASA_AUTO_DEFAULT", STRONG), "empty prompt"
    if len(text) > _env_int("ASA_AUTO_FAST_MAX_CHARS", 400):
        return STRONG, "long prompt"
    if _CODE.search(text):
        return STRONG, "code"
    if _MATH.search(text):
        return STRONG, "math"
    if _HARD.search(text):
        return STRONG, "hard task"
    return FAST, "short simple prompt"


def choose(data: dict) -> str:
    return explain(data)[0]


try:  # Only needed inside the LiteLLM proxy; the heuristic above is importable without it.
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # pragma: no cover
    CustomLogger = object  # type: ignore[misc,assignment]


class AutoRouter(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if isinstance(data, dict) and data.get("model") == AUTO:
            model, reason = explain(data)
            data["model"] = model
            meta = data.setdefault("metadata", {})
            if isinstance(meta, dict):
                meta["asa_auto"] = {"model": model, "reason": reason}
            print(f"[asa-auto] -> {model} ({reason})", flush=True)
        return data


proxy_handler_instance = AutoRouter()
