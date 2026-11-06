#!/usr/bin/env bash
# ASA stage 5: add your own Linux PC to the ASA swarm as a GitHub Actions self-hosted runner.
# Zero cost: GitHub does not charge for self-hosted runner minutes. Labels: self-hosted, asa.
#
# Usage:
#   scripts/asa/self_hosted_runner.sh install [REPO_URL]   # download, register, start as a user service
#   scripts/asa/self_hosted_runner.sh status
#   scripts/asa/self_hosted_runner.sh remove               # stop the service and unregister
#   scripts/asa/self_hosted_runner.sh --help
#
# The registration (or removal) token is pasted interactively and never echoed, written to disk
# or put on a command line (it reaches config.sh through ACTIONS_RUNNER_INPUT_TOKEN, which only
# this user and root can read); get it from
#   GitHub -> repository -> Settings -> Actions -> Runners -> New self-hosted runner.
# It expires after one hour and only works once.
#
# Environment:
#   ASA_RUNNER_DIR      install directory (default ~/actions-runner-asa)
#   ASA_RUNNER_NAME     runner name (default asa-<hostname>)
#   ASA_RUNNER_LABELS   extra labels (default asa; self-hosted/Linux/X64 are added by GitHub)
#   ASA_RUNNER_VERSION  runner version without "v" (default: latest release)
#   ASA_RUNNER_DRY_RUN  1 prints what would run instead of running it (used by the tests)
set -euo pipefail

DIR="${ASA_RUNNER_DIR:-$HOME/actions-runner-asa}"
NAME="${ASA_RUNNER_NAME:-asa-$(hostname -s 2>/dev/null || hostname)}"
LABELS="${ASA_RUNNER_LABELS:-asa}"
DRY="${ASA_RUNNER_DRY_RUN:-0}"
UNIT="asa-runner.service"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
MIN_FREE_GB=60

say() { printf '[asa-runner] %s\n' "$*"; }
die() { printf '[asa-runner] error: %s\n' "$*" >&2; exit 1; }

# Run a command, or print it in dry-run mode. Never pass the token through here.
run() {
    if [ "$DRY" = "1" ]; then printf 'DRY: %s\n' "$*"; else "$@"; fi
}

usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; }

read_secret() {  # read_secret PROMPT -> stdout; no echo on a terminal
    local prompt="$1" value=""
    if [ -t 0 ]; then
        read -r -s -p "$prompt" value
        printf '\n' >&2
    else
        IFS= read -r value || true
    fi
    [ -n "$value" ] || die "empty token"
    printf '%s' "$value"
}

arch() {
    case "$(uname -m)" in
        x86_64|amd64) echo x64 ;;
        aarch64|arm64) echo arm64 ;;
        *) die "unsupported CPU $(uname -m) (need x86_64 or aarch64)" ;;
    esac
}

check_host() {
    [ "$(uname -s)" = "Linux" ] || die "Linux only"
    [ "$(id -u)" != "0" ] || [ "$DRY" = "1" ] || die "do not run as root: the runner executes workflow code as this user"
    for c in curl tar sha256sum; do command -v "$c" >/dev/null || die "missing command: $c"; done
    local free_gb
    free_gb=$(df -Pk "$HOME" | awk 'NR==2 {print int($4/1048576)}')
    if [ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]; then
        say "warning: only ${free_gb} GB free under $HOME; swarm jobs stage expert weights (aim for ${MIN_FREE_GB}+ GB on NVMe)"
    fi
}

latest_version() {
    if [ -n "${ASA_RUNNER_VERSION:-}" ]; then echo "$ASA_RUNNER_VERSION"; return; fi
    curl -fsSL https://api.github.com/repos/actions/runner/releases/latest \
        | sed -n 's/.*"tag_name": *"v\([0-9.]*\)".*/\1/p' | head -n1
}

download() {  # download VERSION ARCH
    local ver="$1" a="$2" file url sha
    file="actions-runner-linux-${a}-${ver}.tar.gz"
    url="https://github.com/actions/runner/releases/download/v${ver}/${file}"
    if [ -x "$DIR/config.sh" ]; then say "runner already unpacked in $DIR"; return; fi
    run mkdir -p "$DIR"
    if [ "$DRY" = "1" ]; then printf 'DRY: download %s and verify sha256\n' "$url"; return; fi
    say "downloading $url"
    curl -fsSL -o "$DIR/$file" "$url"
    # The release notes carry the official checksum between BEGIN/END SHA markers.
    sha=$(curl -fsSL "https://api.github.com/repos/actions/runner/releases/tags/v${ver}" \
        | tr -d '\r' | sed -n "s/.*<!-- BEGIN SHA linux-${a} -->\([0-9a-f]\{64\}\)<!-- END SHA linux-${a} -->.*/\1/p" \
        | head -n1)
    [ -n "$sha" ] || die "could not read the official sha256 for $file; aborting"
    echo "$sha  $DIR/$file" | sha256sum -c - >/dev/null || die "sha256 mismatch for $file"
    tar -xzf "$DIR/$file" -C "$DIR"
    rm -f "$DIR/$file"
    say "verified and unpacked runner v$ver"
}

write_unit() {
    run mkdir -p "$UNIT_DIR"
    local unit
    unit="[Unit]
Description=GitHub Actions runner for the ASA swarm ($NAME)
After=network-online.target

[Service]
WorkingDirectory=$DIR
ExecStart=$DIR/run.sh
Restart=on-failure
RestartSec=10
# Keep the desktop usable while a swarm job streams weights.
Nice=10
IOSchedulingClass=idle

[Install]
WantedBy=default.target
"
    if [ "$DRY" = "1" ]; then printf 'DRY: write %s/%s\n' "$UNIT_DIR" "$UNIT"; else
        printf '%s' "$unit" > "$UNIT_DIR/$UNIT"
    fi
}

have_user_systemd() { command -v systemctl >/dev/null && systemctl --user show-environment >/dev/null 2>&1; }

cmd_install() {
    local url="${1:-}" ver a token
    check_host
    if [ -z "$url" ]; then
        read -r -p "Repository URL (https://github.com/OWNER/REPO): " url
    fi
    [[ "$url" =~ ^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?$ ]] \
        || die "expected https://github.com/OWNER/REPO, got: $url"
    a=$(arch)
    ver=$(latest_version)
    [ -n "$ver" ] || die "could not find the latest runner version (set ASA_RUNNER_VERSION)"
    download "$ver" "$a"
    token=$(read_secret "Registration token (hidden): ")
    say "registering '$NAME' with labels self-hosted,$LABELS"
    if [ "$DRY" = "1" ]; then
        printf 'DRY: ACTIONS_RUNNER_INPUT_TOKEN=*** %s/config.sh --unattended --url %s --name %s --labels %s --work _work --replace\n' \
            "$DIR" "$url" "$NAME" "$LABELS"
    else
        (cd "$DIR" && ACTIONS_RUNNER_INPUT_TOKEN="$token" ./config.sh --unattended --url "$url" \
            --name "$NAME" --labels "$LABELS" --work _work --replace)
    fi
    unset token
    write_unit
    if [ "$DRY" = "1" ] || have_user_systemd; then
        run systemctl --user daemon-reload
        run systemctl --user enable --now "$UNIT"
        say "started as a user service. To keep it running after you log out:"
        say "  sudo loginctl enable-linger $(id -un)"
    else
        say "no systemd user session; start it by hand:  cd $DIR && ./run.sh"
    fi
    say "check it under GitHub -> Settings -> Actions -> Runners (status Idle)."
    say "remove later with: $0 remove"
}

cmd_status() {
    if have_user_systemd; then systemctl --user --no-pager status "$UNIT" || true; fi
    [ -f "$DIR/.runner" ] && say "registered: $(tr -d '\0' < "$DIR/.runner" | sed -n 's/.*"agentName": *"\([^"]*\)".*/\1/p')" \
        || say "not registered in $DIR"
}

cmd_remove() {
    local token
    if [ "$DRY" = "1" ] || have_user_systemd; then
        run systemctl --user disable --now "$UNIT" || true
        run rm -f "$UNIT_DIR/$UNIT"
        run systemctl --user daemon-reload || true
    fi
    if [ "$DRY" != "1" ] && [ ! -x "$DIR/config.sh" ]; then
        say "nothing installed in $DIR"; return
    fi
    say "get a removal token: GitHub -> Settings -> Actions -> Runners -> $NAME -> Remove"
    token=$(read_secret "Removal token (hidden): ")
    if [ "$DRY" = "1" ]; then
        printf 'DRY: ACTIONS_RUNNER_INPUT_TOKEN=*** %s/config.sh remove --unattended\n' "$DIR"
    else
        (cd "$DIR" && ACTIONS_RUNNER_INPUT_TOKEN="$token" ./config.sh remove --unattended)
    fi
    unset token
    say "unregistered. Delete the files with:  rm -rf $DIR"
}

main() {
    local cmd="${1:-}"
    [ $# -gt 0 ] && shift
    case "$cmd" in
        install) cmd_install "$@" ;;
        status) cmd_status ;;
        remove) cmd_remove ;;
        -h|--help|help|"") usage ;;
        *) usage >&2; exit 2 ;;
    esac
}

main "$@"
