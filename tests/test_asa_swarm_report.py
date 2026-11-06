"""asa/swarm/report.py: metrics from a swarm-result artifact (mocked GitHub API)."""

import http.server
import io
import json
import threading
import zipfile

import pytest

from asa.swarm import report as rp

ENGINE = """\
[CLUSTER] coordinator connected to 4 expert worker(s)
[v41] DSpark gate: T(1) 54733.5 ms, rows 3:1.07 4:1.49, rho 0.23, accepted 1.2
[v41] expert I/O: 78 misses, 1466.4 MB in 5.24s (0.28 GB/s); prefill 0.0 MB, disk 0.00s, matmul 0.00s, wall 467.68s
"""
RESULT = {"answer": "There are 92 solutions.", "wall_s": 669.0, "completion_tokens": 8,
          "prompt_tokens": 20, "tok_per_s_end_to_end": 0.012, "timings": None}


def zip_bytes(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, t in files.items():
            z.writestr(n, t)
    return buf.getvalue()


def test_parse_engine_real_lines():
    e = rp.parse_engine(ENGINE)
    assert e == {"expert_io_gb": 1.43, "t1_ms": 54733.5, "connected": 4, "load_s": None}


@pytest.mark.parametrize("line,expect", [
    ("[v41] model loaded in 12.3 s", 12.3),
    ("load time: 4120 ms", 4.1),
    ("[coli] loading weights ... 88.0s", 88.0),
])
def test_parse_load(line, expect):
    assert rp.parse_engine(ENGINE + line + "\n")["load_s"] == expect


def test_no_cluster_and_last_io_wins():
    e = rp.parse_engine("[v41] expert I/O: 1 misses, 10.0 MB in 1s\n"
                        "[v41] expert I/O: 2 misses, 2048.0 MB in 2s\n")
    assert e["connected"] is None and e["expert_io_gb"] == 2.0


def test_metrics_and_row():
    m = rp.metrics(42, {"result.json": json.dumps(RESULT), "engine.err": ENGINE}, 4)
    assert m["cluster_ok"] and m["tok_per_s"] == 0.012 and m["workers"] == 4
    assert rp.row(m) == ("| 42 | 4 | 4 | 669.0 | 8 | 0.012 | – | 54733.5 | 1.43 | "
                         "There are 92 solutions. |")
    m = rp.metrics(1, {"result.json": json.dumps({**RESULT, "tok_per_s_end_to_end": None,
                                                   "answer": "a|b " * 30})}, None)
    assert m["tok_per_s"] == round(8 / 669.0, 3) and not m["cluster_ok"]
    r = rp.row(m)
    import re
    assert "a\\|b" in r and "…" in r
    assert len(re.split(r"(?<!\\)\|", r)) - 2 == len(rp.COLUMNS)  # escaped pipes keep columns
    assert rp.header().splitlines()[0].startswith("| run | workers |")


class FakeGitHub:
    """Serves the API; artifact zip answers 302 to a blob URL that rejects Authorization."""

    def __init__(self, token_required=False):
        self.token_required, self.seen = token_required, []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                auth = self.headers.get("Authorization")
                outer.seen.append((self.path, auth))
                if self.path.startswith("/blob"):
                    if auth:
                        return self._send(401, b"auth header not allowed")
                    return self._send(200, zip_bytes({"result.json": json.dumps(RESULT),
                                                      "engine.err": ENGINE}), "application/zip")
                if outer.token_required and auth != "Bearer tok":
                    return self._send(404, b'{"message":"Not Found"}')
                if "/artifacts?" in self.path:
                    body = {"artifacts": [{"id": 5, "name": "swarm-done"},
                                          {"id": 9, "name": "swarm-result", "expired": False}]}
                elif self.path.endswith("/artifacts/9/zip"):
                    self.send_response(302)
                    self.send_header("Location", f"http://127.0.0.1:{outer.port}/blob/9")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                elif "/jobs?" in self.path:
                    body = {"jobs": [{"name": "plan"}, {"name": "coordinator"}] +
                            [{"name": f"worker ({i})"} for i in range(1, 5)]}
                else:
                    return self._send(404, b'{"message":"Not Found"}')
                self._send(200, json.dumps(body).encode())

            def _send(self, code, body, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_port
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


@pytest.fixture
def gh(monkeypatch):
    g = FakeGitHub()
    monkeypatch.setenv("GITHUB_API_URL", f"http://127.0.0.1:{g.port}")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("ASA_GITHUB_TOKEN", raising=False)
    yield g
    g.srv.shutdown()


def test_report_follows_redirect_without_auth(gh):
    m = rp.report(rp.Api("o/r", token="tok"), 77)
    assert (m["workers"], m["connected"], m["expert_io_gb"]) == (4, 4, 1.43)
    blob = [a for p, a in gh.seen if p.startswith("/blob")]
    assert blob == [None]  # the token never reaches the blob store
    assert all(a == "Bearer tok" for p, a in gh.seen if p.startswith("/repos"))


def test_cli_public_no_token(gh, capsys):
    assert rp.main(["77", "--repo", "o/r"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("| run |") and out[2].startswith("| 77 | 4 | 4 | 669.0 | 8 |")
    assert all(a is None for _, a in gh.seen)
    assert rp.main(["77", "--repo", "o/r", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["t1_ms"] == 54733.5


def test_cli_missing_artifact(gh, capsys, monkeypatch):
    gh.token_required = True  # private repo without a token -> 404
    assert rp.main(["77", "--repo", "o/r"]) == 1
    assert "run 77" in capsys.readouterr().err
