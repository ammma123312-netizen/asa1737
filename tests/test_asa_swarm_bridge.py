"""asa/swarm/bridge.py: the asynchronous asa-swarm model for the app."""

import json
import os
import stat
import subprocess
import sys
import textwrap
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from asa.swarm import bridge as br
from asa.swarm import issue_client as ic

ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self):
        self.asked, self.answers, self.next_issue = [], {}, 41

    def ask(self, prompt):
        self.next_issue += 1
        self.asked.append(prompt)
        return self.next_issue

    def answer(self, n):
        a = self.answers.get(n)
        if isinstance(a, Exception):
            raise a
        return ic.Answer(n, a, "u") if a else None


@pytest.fixture
def setup(tmp_path):
    client, now = FakeClient(), [1000.0]
    b = br.Bridge(client, br.JobStore(str(tmp_path / "jobs.json")), "o/r", clock=lambda: now[0])
    return b, client, now


def user(text):
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": text}]


def test_submit_then_status_then_answer(setup):
    b, client, now = setup
    r = b.reply(user("How many 8-queens solutions?"))
    assert "issue #42" in r and "https://github.com/o/r/issues/42" in r and "posted as an issue" in r
    # Same prompt again (whitespace-insensitive) does not open a second issue.
    now[0] += 600
    r = b.reply(user("  How many   8-queens solutions? "))
    assert "Still running" in r and "10 min ago" in r and client.asked == ["How many 8-queens solutions?"]
    client.answers[42] = "92"
    assert b.reply(user("How many 8-queens solutions?")) == "92"
    assert b.reply(user("status")) == "92"
    assert b.reply(user("#42")) == "92"
    assert b.reply(user("حالة #42")) == "92"
    assert b.reply(user("#7")) == "No asa-swarm job for issue #7."


def test_failure_is_reported_and_resubmits(setup):
    b, client, _ = setup
    b.reply(user("q"))
    client.answers[42] = ic.SwarmFailed(42, "ASA swarm run failed", "https://x/run")
    r = b.reply(user("q"))
    assert "failed" in r and "https://x/run" in r and "again to retry" in r
    assert "issue #43" in b.reply(user("q"))
    assert client.asked == ["q", "q"]


def test_api_errors_and_unconfigured(setup, tmp_path):
    b, client, _ = setup
    client.ask = lambda p: (_ for _ in ()).throw(ic.SwarmError("401 Bad credentials"))
    assert "Could not submit" in b.reply(user("q"))
    assert "Send a question" in b.reply(user("  "))
    assert "No asa-swarm jobs yet" in b.reply(user("status"))
    nb = br.Bridge(None, br.JobStore(str(tmp_path / "j2.json")), "")
    assert "ASA_GITHUB_TOKEN" in nb.reply(user("q"))


def test_job_store_persists(tmp_path):
    p = str(tmp_path / "jobs.json")
    br.JobStore(p).add("hello", 5, 1.0)
    s = br.JobStore(p)
    assert s.get("hello")["issue"] == 5 and s.by_issue(5) and s.latest()["issue"] == 5
    s.drop_issue(5)
    assert s.get("hello") is None


def test_key_file_private(monkeypatch, tmp_path):
    monkeypatch.setenv("ASA_SWARM_KEY_FILE", str(tmp_path / "k"))
    k = br.ensure_key()
    assert k.startswith("sk-swarm-") and br.ensure_key() == k
    assert stat.S_IMODE((tmp_path / "k").stat().st_mode) == 0o600


@pytest.fixture
def server(setup):
    b, client, _ = setup
    srv = br.http.server.ThreadingHTTPServer(("127.0.0.1", 0), br.make_handler(b, "sk-test"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", client
    srv.shutdown()


def _post(url, body, key="sk-test"):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.headers.get("Content-Type"), r.read().decode()


def test_http_openai_compatible(server):
    base, client = server
    ctype, body = _post(base + "/v1/chat/completions", {"model": "asa-swarm", "messages": user("q")})
    assert ctype == "application/json"
    assert "issue #42" in json.loads(body)["choices"][0]["message"]["content"]
    client.answers[42] = "answer text"
    ctype, body = _post(base + "/v1/chat/completions",
                        {"model": "asa-swarm", "stream": True, "messages": user("q")})
    assert ctype == "text/event-stream"
    events = [l[6:] for l in body.splitlines() if l.startswith("data: ")]
    assert events[-1] == "[DONE]"
    assert json.loads(events[0])["choices"][0]["delta"]["content"] == "answer text"
    req = urllib.request.Request(base + "/v1/models", headers={"Authorization": "Bearer sk-test"})
    with urllib.request.urlopen(req, timeout=10) as r:
        assert [m["id"] for m in json.load(r)["data"]] == ["asa-swarm"]


def test_http_rejects_bad_key(server):
    base, client = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base + "/v1/chat/completions", {"messages": user("q")}, key="wrong")
    assert e.value.code == 401 and client.asked == []


def test_disabled_is_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("ASA_SWARM", raising=False)
    monkeypatch.setenv("ASA_SWARM_KEY_FILE", str(tmp_path / "k"))
    assert br.main(["serve"]) == 0 and not (tmp_path / "k").exists()


def test_register(tmp_path):
    try:
        import core.database  # noqa: F401  (needs the full app: fastapi, httpx, ...)
    except Exception:
        pytest.skip("core.database needs the full app dependencies")
    script = textwrap.dedent("""
        import json
        from asa.swarm import bridge as br
        out = [br.register("sk-a"), br.register("sk-a"), br.register("sk-b")]
        from core.database import SessionLocal, ModelEndpoint
        r = SessionLocal().query(ModelEndpoint).one()
        out.append([r.base_url, r.api_key, json.loads(r.pinned_models), r.supports_tools])
        print(json.dumps(out))
    """)
    env = dict(os.environ, DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}", PYTHONPATH=str(ROOT))
    env.pop("ASA_SWARM_PORT", None)
    res = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr[-2000:]
    assert json.loads(res.stdout.strip().splitlines()[-1]) == [
        "created", "unchanged", "key-updated",
        ["http://127.0.0.1:4100/v1", "sk-b", ["asa-swarm"], False]]


def test_entrypoint_hook():
    text = (ROOT / "docker" / "entrypoint.sh").read_text()
    i = text.index('if [ "${ASA_SWARM:-0}" = "1" ]')
    assert i < text.index("python -m asa.swarm.bridge register") \
        < text.index("python -m asa.swarm.bridge serve) &") < text.index('exec gosu "$PUID:$PGID" "$@"')
