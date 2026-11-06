"""ASA stage 5: scripts/asa/self_hosted_runner.sh (syntax, flags, token handling) in dry-run mode."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "asa" / "self_hosted_runner.sh"
URL = "https://github.com/o/r"
TOKEN = "AABBCCDD-secret-registration-token"


def run(args, tmp_path, stdin="", **env):
    e = dict(os.environ, HOME=str(tmp_path), ASA_RUNNER_DRY_RUN="1", ASA_RUNNER_VERSION="2.329.0",
             ASA_RUNNER_DIR=str(tmp_path / "runner"), ASA_RUNNER_NAME="asa-test",
             XDG_CONFIG_HOME=str(tmp_path / "cfg"), **env)
    return subprocess.run(["bash", str(SCRIPT), *args], input=stdin, env=e,
                          capture_output=True, text=True, timeout=60)


def test_bash_syntax():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    assert os.access(SCRIPT, os.X_OK)


@pytest.mark.skipif(not shutil.which("shellcheck"), reason="shellcheck not installed")
def test_shellcheck():
    res = subprocess.run(["shellcheck", "-S", "warning", str(SCRIPT)], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout


def test_help_and_unknown(tmp_path):
    r = run(["--help"], tmp_path)
    assert r.returncode == 0 and "install [REPO_URL]" in r.stdout and "remove" in r.stdout
    assert run(["bogus"], tmp_path).returncode == 2


@pytest.mark.skipif(os.uname().sysname != "Linux", reason="Linux only script")
def test_install_dry_run_flags_and_token_never_printed(tmp_path):
    r = run(["install", URL], tmp_path, stdin=TOKEN + "\n")
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    assert TOKEN not in out
    assert "releases/download/v2.329.0/actions-runner-linux-" in out and "verify sha256" in out
    cfg = next(l for l in out.splitlines() if "config.sh --unattended" in l)
    assert "ACTIONS_RUNNER_INPUT_TOKEN=***" in cfg and "--token" not in cfg
    for flag in (f"--url {URL}", "--name asa-test", "--labels asa", "--work _work", "--replace"):
        assert flag in cfg
    assert "systemctl --user enable --now asa-runner.service" in out
    assert "loginctl enable-linger" in out
    # Nothing written in dry-run mode, and never a file containing the token.
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert TOKEN not in p.read_text(errors="ignore")


def test_install_rejects_bad_url_and_empty_token(tmp_path):
    r = run(["install", "http://evil.example/o/r"], tmp_path, stdin=TOKEN)
    assert r.returncode != 0 and "expected https://github.com/OWNER/REPO" in r.stderr
    r = run(["install", URL], tmp_path, stdin="\n")
    assert r.returncode != 0 and "empty token" in r.stderr


def test_remove_dry_run(tmp_path):
    r = run(["remove"], tmp_path, stdin=TOKEN + "\n")
    assert r.returncode == 0, r.stderr
    assert TOKEN not in r.stdout + r.stderr
    assert "systemctl --user disable --now asa-runner.service" in r.stdout
    assert "ACTIONS_RUNNER_INPUT_TOKEN=*** " in r.stdout and "config.sh remove --unattended" in r.stdout


def test_token_only_via_env_in_real_commands():
    text = SCRIPT.read_text()
    assert 'read -r -s -p' in text  # hidden prompt on a terminal
    assert '--token "$token"' not in text and "> \"$token\"" not in text
    assert text.count('ACTIONS_RUNNER_INPUT_TOKEN="$token"') == 2
    assert 'die "do not run as root' in text
