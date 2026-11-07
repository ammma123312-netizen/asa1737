"""asa.swarm.ask: streamed completion, per-token timestamps, deadline, partial file."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from asa.swarm import ask

PIECES = ["عدد ", "الحلول ", "هو ", "92"]


def serve(delay=0.0):
    seen = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"data":[]}')

        def do_POST(self):
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for i, p in enumerate(PIECES):
                    delta = {"reasoning_content": "fikr"} if i == 0 and seen["body"].get("enable_thinking") else {}
                    delta["content"] = p
                    self.wfile.write(f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(delay)
                end = {"choices": [{"delta": {}, "finish_reason": "stop"}],
                       "usage": {"completion_tokens": 4, "prompt_tokens": 9}}
                self.wfile.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1", seen


def test_stream_collects_tokens_with_times(tmp_path):
    srv, base, seen = serve()
    part = tmp_path / "partial.json"
    res = ask.ask(base, "k", "m", "q", 50, partial=str(part))
    srv.shutdown()
    assert seen["body"]["stream"] is True and "enable_thinking" not in seen["body"]
    assert seen["body"]["messages"] == [{"role": "user", "content": "q"}]  # no system prompt added
    assert res["answer"] == "".join(PIECES) and res["completion_tokens"] == 4
    assert res["finish_reason"] == "stop" and res["stopped"] is None
    assert [x["text"] for x in res["tokens"]] == PIECES
    assert json.loads(part.read_text())["answer"] == "".join(PIECES)


def test_deadline_keeps_partial(tmp_path):
    srv, base, _ = serve(delay=0.3)
    part = tmp_path / "partial.json"
    res = ask.ask(base, "k", "m", "q", 50, deadline_s=0.1, partial=str(part))
    srv.shutdown()
    assert res["stopped"] == "deadline"
    assert 1 <= len(res["tokens"]) < len(PIECES)
    assert json.loads(part.read_text())["answer"] == res["answer"]


def test_thinking_flag_and_reasoning(tmp_path):
    srv, base, seen = serve()
    res = ask.ask(base, "k", "m", "q", 50, thinking=True)
    srv.shutdown()
    assert seen["body"]["enable_thinking"] is True
    assert res["reasoning"] == "fikr" and res["tokens"][0]["kind"] == "reasoning"


def test_main_writes_result(tmp_path):
    srv, base, _ = serve()
    pf = tmp_path / "p.txt"; pf.write_text("q", encoding="utf-8")
    out = tmp_path / "r.json"
    assert ask.main(["--base", base, "--key", "k", "--prompt-file", str(pf), "--out", str(out)]) == 0
    srv.shutdown()
    r = json.loads(out.read_text())
    assert r["answer"].endswith("92") and "ready_s" in r and len(r["tokens"]) == 4
