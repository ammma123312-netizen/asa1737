#!/bin/sh
# Start the ASA router (LiteLLM proxy with asa/router/litellm.yaml) inside the Odysseus container.
# Stage 6 of docs/asa/README.md. Started in the background by docker/entrypoint.sh.
#
# Environment:
#   ASA_ROUTER            1 to enable (default off)
#   ASA_ROUTER_PORT       port on 127.0.0.1 (default 4000)
#   ASA_DEEPSEEK_BASE     OpenAI base URL of Colibri / the swarm coordinator
#                         (default http://127.0.0.1:8000/v1)
#   LITELLM_MASTER_KEY    optional; otherwise read from data/asa_router.key (see seed_endpoint.py)
#   ASA_ROUTER_INSTALL    1 (default) installs litellm[proxy] with pip --user if it is missing
set -e

if [ "${ASA_ROUTER:-0}" != "1" ]; then
    exit 0
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="${ASA_ROUTER_PORT:-4000}"
export ASA_DEEPSEEK_BASE="${ASA_DEEPSEEK_BASE:-http://127.0.0.1:8000/v1}"

LITELLM_MASTER_KEY="$(python "$HERE/seed_endpoint.py" --print-key)"
export LITELLM_MASTER_KEY

if ! command -v litellm >/dev/null 2>&1; then
    if [ "${ASA_ROUTER_INSTALL:-1}" = "1" ]; then
        echo "[asa-router] installing litellm[proxy] (first start only)..."
        python -m pip install --user --quiet "litellm[proxy]" || {
            echo "[asa-router] pip install failed; router not started" >&2
            exit 0
        }
        PATH="$HOME/.local/bin:$PATH"
    else
        echo "[asa-router] litellm not installed and ASA_ROUTER_INSTALL=0; router not started" >&2
        exit 0
    fi
fi

echo "[asa-router] LiteLLM on 127.0.0.1:$PORT -> asa-deepseek at $ASA_DEEPSEEK_BASE"
# 127.0.0.1 only: the app is the sole client; public traffic goes through Odysseus.
exec litellm --config "$HERE/litellm.yaml" --host 127.0.0.1 --port "$PORT"
