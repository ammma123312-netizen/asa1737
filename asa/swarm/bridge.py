"""``asa-swarm`` for the Odysseus app: an OpenAI-compatible bridge to the issue-based swarm.

A swarm answer takes many minutes, far longer than a chat request may block, so the bridge is
asynchronous and needs no UI changes:

  1. A new message opens an ``asa-ask`` issue (issue_client.ask) and replies at once with
     "submitted as issue #N".
  2. Sending the same message again, ``status``, ``#N`` or ``حالة`` replies with the answer
     when it has arrived, or with "still running".
  3. If the run failed, the reply says so with the run link, and the job is forgotten, so
     sending the message again submits it again.

Off by default (``ASA_SWARM=1`` enables it in docker/entrypoint.sh). Listens on 127.0.0.1 only
and requires a local key (``data/asa_swarm.key``, mode 0600) that ``register`` stores in the
app's encrypted endpoint row. The GitHub token is read only from the environment
(``ASA_GITHUB_TOKEN``, else ``GITHUB_TOKEN``); it is never stored by this module, never put in
replies and never logged. Prompts are posted as issues in ``ASA_SWARM_REPO``: replies say so.

Run:  python -m asa.swarm.bridge serve      (ASA_SWARM_PORT, default 4100)
      python -m asa.swarm.bridge register   (adds the endpoint to the app's DB)
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import re
import secrets
import sys
import threading
import time
import uuid
from typing import Optional

from asa.swarm import issue_client as ic

MODEL = "asa-swarm"
ENDPOINT_NAME = "ASA Swarm (async, GitHub issues)"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

_STATUS_WORDS = {"status", "?", "؟", "حالة", "الحالة", "النتيجة", "result"}
_ISSUE_REF = re.compile(r"^\s*(?:status|حالة|الحالة)?\s*#(\d+)\s*$", re.IGNORECASE)


def enabled() -> bool:
    return os.getenv("ASA_SWARM", "0") == "1"


def port() -> int:
    return int(os.getenv("ASA_SWARM_PORT", "4100"))


def base_url() -> str:
    return f"http://127.0.0.1:{port()}/v1"


def _private_file(path: str, make) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            value = f.read().strip()
        if value:
            return value
    except FileNotFoundError:
        pass
    value = make()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(value + "\n")
    return value


def ensure_key() -> str:
    return _private_file(os.getenv("ASA_SWARM_KEY_FILE") or os.path.join(DATA_DIR, "asa_swarm.key"),
                         lambda: "sk-swarm-" + secrets.token_hex(16))


class JobStore:
    """prompt hash -> {issue, created}; a JSON file so jobs survive restarts."""

    def __init__(self, path: str):
        self.path, self._lock = path, threading.Lock()

    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, jobs: dict) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(jobs, f)
        os.replace(tmp, self.path)

    @staticmethod
    def key(prompt: str) -> str:
        return hashlib.sha256(" ".join(prompt.split()).encode()).hexdigest()

    def get(self, prompt: str) -> Optional[dict]:
        return self._load().get(self.key(prompt))

    def by_issue(self, issue: int) -> Optional[dict]:
        return next((j for j in self._load().values() if j["issue"] == issue), None)

    def latest(self) -> Optional[dict]:
        jobs = list(self._load().values())
        return max(jobs, key=lambda j: j["created"]) if jobs else None

    def add(self, prompt: str, issue: int, now: float) -> dict:
        with self._lock:
            jobs = self._load()
            job = jobs[self.key(prompt)] = {"issue": issue, "created": now}
            self._save(jobs)
            return job

    def drop_issue(self, issue: int) -> None:
        with self._lock:
            jobs = {k: v for k, v in self._load().items() if v["issue"] != issue}
            self._save(jobs)


class Bridge:
    def __init__(self, client: Optional[ic.SwarmIssueClient], store: JobStore, repo: str,
                 clock=time.time):
        self.client, self.store, self.repo, self.clock = client, store, repo, clock

    def _issue_url(self, n: int) -> str:
        return f"https://github.com/{self.repo}/issues/{n}"

    def _report(self, job: dict) -> str:
        n = job["issue"]
        try:
            got = self.client.answer(n)
        except ic.SwarmFailed as e:
            self.store.drop_issue(n)
            return (f"The ASA swarm run for issue #{n} failed ({e.comment_url or self._issue_url(n)}). "
                    "Send the same message again to retry.")
        except ic.SwarmError as e:
            return f"Could not check issue #{n} right now: {e}"
        if got is None:
            mins = max(0, int((self.clock() - job["created"]) // 60))
            return (f"Still running on the ASA swarm: issue #{n} ({self._issue_url(n)}), "
                    f"started {mins} min ago. Send `status` or `#{n}` to check again.")
        return got.text

    def reply(self, messages: list) -> str:
        if self.client is None:
            return ("asa-swarm is not configured: set ASA_SWARM_REPO (owner/name) and "
                    "ASA_GITHUB_TOKEN (a token with Issues: read and write) and restart.")
        last = next((m for m in reversed(messages or [])
                     if isinstance(m, dict) and m.get("role") == "user"), None)
        text, _ = ic_text(last.get("content") if last else "")
        text = text.strip()
        if not text:
            return "Send a question for the ASA swarm."
        ref = _ISSUE_REF.match(text)
        if ref:
            job = self.store.by_issue(int(ref.group(1)))
            return self._report(job) if job else f"No asa-swarm job for issue #{ref.group(1)}."
        if text.lower() in _STATUS_WORDS:
            job = self.store.latest()
            return self._report(job) if job else "No asa-swarm jobs yet. Send a question first."
        job = self.store.get(text)
        if job:
            return self._report(job)
        try:
            n = self.client.ask(text)
        except ic.SwarmError as e:
            return f"Could not submit to the ASA swarm: {e}"
        self.store.add(text, n, self.clock())
        return (f"Submitted to the ASA swarm (DeepSeek-V4.1-Flash 763B on free GitHub runners) as "
                f"issue #{n}: {self._issue_url(n)}\n\nRuns usually take 10 to 60 minutes. Send the "
                f"same message again, `status` or `#{n}` to get the answer.\n\nNote: the question "
                f"is posted as an issue in {self.repo}; anyone who can see that repository can "
                f"read it.")


def ic_text(content) -> tuple[str, bool]:
    if isinstance(content, str):
        return content, False
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content
                         if isinstance(p, dict) and p.get("type") == "text"), True
    return "", False


def _completion(text: str, stream: bool):
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
    if not stream:
        return json.dumps({
            "id": cid, "object": "chat.completion", "created": created, "model": MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }).encode()
    chunk = lambda delta, fin: "data: " + json.dumps({
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}) + "\n\n"
    return (chunk({"role": "assistant", "content": text}, None) + chunk({}, "stop")
            + "data: [DONE]\n\n").encode()


def make_handler(bridge: Bridge, key: str):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # no request logging: paths only, never headers
            pass

        def _send(self, code: int, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authed(self) -> bool:
            got = self.headers.get("Authorization", "")
            if secrets.compare_digest(got.encode(), f"Bearer {key}".encode()):
                return True
            self._send(401, b'{"error":{"message":"invalid api key"}}')
            return False

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                if self._authed():
                    self._send(200, json.dumps({"object": "list", "data": [
                        {"id": MODEL, "object": "model", "owned_by": "asa"}]}).encode())
            elif self.path == "/health":
                self._send(200, b'{"ok":true}')
            else:
                self._send(404, b'{"error":{"message":"not found"}}')

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                return self._send(404, b'{"error":{"message":"not found"}}')
            if not self._authed():
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            except (ValueError, json.JSONDecodeError):
                return self._send(400, b'{"error":{"message":"invalid json"}}')
            stream = bool(req.get("stream"))
            body = _completion(bridge.reply(req.get("messages") or []), stream)
            self._send(200, body, "text/event-stream" if stream else "application/json")

    return Handler


def bridge_from_env(store_path: Optional[str] = None) -> Bridge:
    repo = os.getenv("ASA_SWARM_REPO", "")
    store = JobStore(store_path or os.path.join(DATA_DIR, "asa_swarm_jobs.json"))
    try:
        client = ic.from_env()
    except ValueError:
        client = None
    return Bridge(client, store, repo)


def register(key: str) -> str:
    sys.path.insert(0, BASE_DIR)
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{os.path.join(DATA_DIR, 'app.db')}")
    from core.database import Base, ModelEndpoint, SessionLocal, engine

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        row = db.query(ModelEndpoint).filter(ModelEndpoint.base_url == base_url()).one_or_none()
        if row is None:
            db.add(ModelEndpoint(
                id=str(uuid.uuid4()), name=ENDPOINT_NAME, base_url=base_url(), api_key=key,
                is_enabled=True, pinned_models=json.dumps([MODEL]), cached_models=json.dumps([MODEL]),
                model_type="llm", endpoint_kind="local", model_refresh_mode="auto",
                supports_tools=False, owner=None))
            db.commit()
            return "created"
        if row.api_key != key:
            row.api_key = key
            db.commit()
            return "key-updated"
        return "unchanged"
    finally:
        db.close()


def main(argv: list[str]) -> int:
    if not enabled():
        return 0
    cmd = argv[0] if argv else "serve"
    key = ensure_key()
    if cmd == "register":
        try:
            print(f"[asa-swarm] endpoint at {base_url()}: {register(key)}")
        except Exception as exc:  # never block container start
            print(f"[asa-swarm] could not register endpoint: {exc}", file=sys.stderr)
        return 0
    bridge = bridge_from_env()
    if bridge.client is None:
        print("[asa-swarm] ASA_SWARM_REPO / ASA_GITHUB_TOKEN not set: replies will explain how "
              "to configure it", file=sys.stderr)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port()), make_handler(bridge, key))
    print(f"[asa-swarm] bridge on {base_url()} -> issues in {bridge.repo or '(unset)'}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
