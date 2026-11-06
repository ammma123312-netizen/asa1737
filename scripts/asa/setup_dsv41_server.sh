#!/usr/bin/env bash
# ASA: set up DeepSeek-V4.1-Flash (763B, MIT) at full original quality on one Linux server.
#
# Why this model: it is the 700B-class open model that streams fastest from disk at
# native precision. Experts ship in FP4 and the rest in FP8, and Colibrì reads the
# released checkpoint as-is (no conversion, token-exact), reading ~4.5 GB of experts
# per token vs ~12.7 GB for GLM-5.2. See docs/asa/README.md sections 10-11.
#
# Requirements: Linux x86-64, >= 32 GB RAM (64 GB+ recommended), >= 520 GB free on a
# fast NVMe for MODEL_DIR. A second NVMe for MIRROR_DIR roughly adds its bandwidth.
#
# Usage (as root or with sudo for packages/service):
#   MODEL_DIR=/nvme1/dsv41 MIRROR_DIR=/nvme2/dsv41 ./scripts/asa/setup_dsv41_server.sh
#
# Environment:
#   MODEL_DIR      where the 510 GB checkpoint lives      (default /srv/models/DeepSeek-V4.1-Flash)
#   MIRROR_DIR     optional second drive; gets a copy of the expert shards (partial mirror)
#   REPO           Hugging Face repo                     (default deepseek-ai/DeepSeek-V4.1-Flash)
#   COLIBRI_DIR    where to build Colibrì                 (default /opt/colibri)
#   COLIBRI_REF    pinned commit                          (default below)
#   PORT, MODEL_ID OpenAI endpoint port and model id      (default 8000, deepseek-v4.1-flash)
#   RAM_GB         RAM budget for the expert cache        (default: let Colibrì plan)
#   DOWNLOAD       full | meta | skip                      (meta = config/tokenizer only, for testing)
#   INSTALL_SERVICE 1 to install a systemd unit            (default 1 when run as root)
set -euo pipefail

REPO="${REPO:-deepseek-ai/DeepSeek-V4.1-Flash}"
MODEL_DIR="${MODEL_DIR:-/srv/models/DeepSeek-V4.1-Flash}"
MIRROR_DIR="${MIRROR_DIR:-}"
COLIBRI_DIR="${COLIBRI_DIR:-/opt/colibri}"
COLIBRI_REF="${COLIBRI_REF:-0a5504a1dc64f86fcd0fe7db298b0448c730e7e6}"
PORT="${PORT:-8000}"
MODEL_ID="${MODEL_ID:-deepseek-v4.1-flash}"
DOWNLOAD="${DOWNLOAD:-full}"
VENV="${VENV:-$COLIBRI_DIR/.venv-asa}"
if [ "$(id -u)" = "0" ]; then INSTALL_SERVICE="${INSTALL_SERVICE:-1}"; else INSTALL_SERVICE="${INSTALL_SERVICE:-0}"; fi
NEED_GB=520

log() { printf '\n[asa] %s\n' "$*"; }
die() { printf '[asa] ERROR: %s\n' "$*" >&2; exit 1; }

# 1) preflight ---------------------------------------------------------------
log "preflight"
ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
echo "  RAM: ${ram_gb} GB · CPUs: $(nproc)"
[ "$ram_gb" -ge 30 ] || echo "  WARNING: <32 GB RAM; dense weights (~18 GB) leave little room for the expert cache"
mkdir -p "$MODEL_DIR"
free_gb=$(df --output=avail -BG "$MODEL_DIR" | tail -1 | tr -dc 0-9)
echo "  free on $(df --output=target "$MODEL_DIR" | tail -1): ${free_gb} GB"
if [ "$DOWNLOAD" = "full" ]; then
    have_gb=$(du -s -BG "$MODEL_DIR" 2>/dev/null | cut -f1 | tr -dc 0-9)
    [ $((free_gb + ${have_gb:-0})) -ge "$NEED_GB" ] || die "need ~${NEED_GB} GB for $REPO, have ${free_gb} GB free"
fi

# 2) build tools -------------------------------------------------------------
log "build dependencies"
if ! command -v gcc >/dev/null || ! command -v make >/dev/null || ! command -v git >/dev/null; then
    [ "$(id -u)" = "0" ] || die "install gcc, make, git (and libgomp) first, or rerun as root"
    apt-get update -qq && apt-get install -y -qq build-essential git python3-venv
fi

log "Colibrì @ ${COLIBRI_REF:0:7} -> $COLIBRI_DIR"
if [ ! -d "$COLIBRI_DIR/.git" ]; then
    git clone -q https://github.com/JustVugg/colibri "$COLIBRI_DIR"
fi
git -C "$COLIBRI_DIR" fetch -q origin "$COLIBRI_REF" 2>/dev/null || true
git -C "$COLIBRI_DIR" checkout -q "$COLIBRI_REF"
(cd "$COLIBRI_DIR/c" && ./setup.sh >/dev/null && make -s deepseek_v41)
echo "  built: $COLIBRI_DIR/c/deepseek_v41"

log "python env for download + engram preparation"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade "huggingface_hub[hf_xet]" transformers tokenizers numpy sympy

# 3) weights -----------------------------------------------------------------
log "download ($DOWNLOAD) $REPO -> $MODEL_DIR"
case "$DOWNLOAD" in
    full) "$VENV/bin/python" - "$REPO" "$MODEL_DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], max_workers=16)  # resumes if interrupted
EOF
    ;;
    meta) "$VENV/bin/python" - "$REPO" "$MODEL_DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], ignore_patterns=["*.safetensors", "*.pdf", "assets/*"])
EOF
    ;;
    skip) echo "  skipped" ;;
    *) die "DOWNLOAD must be full, meta or skip" ;;
esac

log "engram sidecar (once per checkpoint)"
(cd "$COLIBRI_DIR" && "$VENV/bin/python" c/tools/prepare_dsv41.py --model "$MODEL_DIR")

# 4) optional second drive ---------------------------------------------------
if [ -n "$MIRROR_DIR" ] && [ "$DOWNLOAD" = "full" ]; then
    log "partial mirror of expert shards -> $MIRROR_DIR"
    mkdir -p "$MIRROR_DIR"
    (cd "$COLIBRI_DIR/c" && ./coli mirror plan  --model "$MODEL_DIR" --mirror "$MIRROR_DIR" --reserve-gib 20 \
        && ./coli mirror stage --model "$MODEL_DIR" --mirror "$MIRROR_DIR" --reserve-gib 20 \
        && ./coli mirror verify --model "$MODEL_DIR" --mirror "$MIRROR_DIR") \
        || echo "  mirror step failed; serving from the primary drive only"
fi

# 5) readiness ---------------------------------------------------------------
log "doctor"
(cd "$COLIBRI_DIR/c" && ./coli doctor --model "$MODEL_DIR") || true

# 6) service -----------------------------------------------------------------
ram_flag=""; [ -n "${RAM_GB:-}" ] && ram_flag="--ram $RAM_GB"
cmd="$COLIBRI_DIR/c/coli serve --model $MODEL_DIR --model-id $MODEL_ID --host 127.0.0.1 --port $PORT $ram_flag"
if [ "$INSTALL_SERVICE" = "1" ] && [ "$DOWNLOAD" = "full" ]; then
    log "systemd unit asa-dsv41.service"
    cat > /etc/systemd/system/asa-dsv41.service <<EOF
[Unit]
Description=ASA DeepSeek-V4.1-Flash (Colibri, native precision)
After=network-online.target

[Service]
Environment=COLI_MODEL_MIRROR=${MIRROR_DIR}
Environment=COLI_API_KEY=${COLI_API_KEY:-}
ExecStart=$cmd
Restart=on-failure
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload && systemctl enable --now asa-dsv41.service
    echo "  started; logs: journalctl -u asa-dsv41 -f"
else
    log "start manually:"
    echo "  COLI_MODEL_MIRROR=$MIRROR_DIR $cmd"
fi

cat <<EOF

[asa] done. OpenAI endpoint: http://127.0.0.1:$PORT/v1  (model: $MODEL_ID)
      Benchmark:  python -m asa.bench_api --port $PORT --model $MODEL_ID
      Router:     litellm --config asa/router/litellm.yaml --port 4000
EOF
