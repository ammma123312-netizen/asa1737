#!/bin/sh
# Entrypoint that fixes the #1 self-host footgun: a Docker container
# that runs as root writes root-owned files into bind-mounted host
# volumes, and the host user (or a non-root service user) then can't
# update them — silently breaking skill extraction, prefs saves, mail
# attachments, etc.
#
# Standard PUID/PGID pattern: pick the UID/GID we should drop to,
# chown the writable bind-mounts so existing root-owned content gets
# repaired on every start (idempotent), then exec the real command
# as that user via gosu.
set -e

PUID="${PUID:-1000}"
PGID="${PGID:-1000}"

# Reuse an existing matching group/user if the host's UID/GID already
# corresponds to one in /etc/passwd (e.g. when the image is rebuilt
# and "odysseus" already exists at the same id). Otherwise create.
if ! getent group "$PGID" >/dev/null 2>&1; then
    groupadd -g "$PGID" odysseus
fi
if ! getent passwd "$PUID" >/dev/null 2>&1; then
    useradd -u "$PUID" -g "$PGID" -M -s /bin/sh -d /app odysseus
fi

# Repair ownership on every writable path the app touches at runtime.
#
# Bind-mounted dirs (/app/data, /app/logs) are the obvious ones, but
# the app ALSO writes inside the image's own source tree at runtime:
#   - services/cache/{search,content}/*  (search cache LRU)
#   - services/search_analytics.json
#   - services/search_engine_error.log
#   - services/tts cache, etc.
# These dirs were created as root during `docker build`, so dropping
# to PUID:PGID would otherwise crash on the first import that tries
# to mkdir them. Chown the whole /app tree — fast (<1s on this size)
# and idempotent via the `-not -uid` filter so we only touch files
# that need fixing.
for dir in /app /app/data /app/logs; do
    if [ -d "$dir" ]; then
        # `find ... -not -uid` keeps this O(touched-files), not
        # O(everything), so terabyte-sized maildirs don't slow startup.
        find "$dir" -not -uid "$PUID" -print0 2>/dev/null \
            | xargs -0 -r chown "$PUID:$PGID" 2>/dev/null || true
    fi
done

# Cookbook installs vllm/etc. via `pip install --user`, which pulls
# nvidia-cuda-* wheels into /app/.local but does not set CUDA_HOME or
# symlink /usr/local/cuda. vllm 0.22+ then crashes during engine init
# when FlashInfer tries to JIT a sampler kernel ("Could not find nvcc",
# then "CUDA compiler and toolkit headers are incompatible" on the
# mixed cuda-nvcc 13.3 / cuda-runtime 13.0 wheel combo).
#
# Auto-set CUDA_HOME if a pip-installed nvcc is present, and disable the
# FlashInfer JIT sampler — sampler only, no impact on attention path.
# No-op when vllm isn't installed.
#
# Checked layouts (all are real pip-wheel install paths):
#   nvidia/cu13        — nvidia-nvcc-cu13 (CUDA 13.x wheel style)
#   nvidia/cu12        — nvidia-nvcc-cu12 (CUDA 12.x wheel style)
#   nvidia/cuda_nvcc   — nvidia-cuda-nvcc-cu12 (older cu12 sub-package style)
for cu in \
    /app/.local/lib/python*/site-packages/nvidia/cu13 \
    /app/.local/lib/python*/site-packages/nvidia/cu12 \
    /app/.local/lib/python*/site-packages/nvidia/cuda_nvcc; do
    if [ -x "$cu/bin/nvcc" ]; then
        export CUDA_HOME="$cu"
        break
    fi
done
# Disable the FlashInfer JIT sampler unconditionally — it is sampler-only
# and has no impact on the attention path, but requires nvcc + matching
# CUDA headers at startup. Without this, vLLM crashes with "Could not find
# nvcc" even when the GPU itself is fully visible to the container.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# Make Cookbook-installed Python CLIs visible after `pip install --user`.
# vLLM and helper scripts land here because /app is the non-root user's HOME.
export PATH="/app/.local/bin:$PATH"

# HF Spaces persistence: pull last snapshot from companion HF Dataset BEFORE
# setup.py runs (so existing app.db is detected and setup is skipped).
# No-op when HF_TOKEN / PERSIST_REPO_ID aren't set — preserves vanilla Docker
# behavior outside HF Spaces.
if [ -n "$HF_TOKEN" ] && [ -n "$PERSIST_REPO_ID" ]; then
    echo "[entrypoint] pulling persistence snapshot from $PERSIST_REPO_ID..."
    gosu "$PUID:$PGID" python /app/scripts/hf_persistence.py pull || \
        echo "[entrypoint] pull failed (non-fatal, continuing with empty data)"
fi

# Initialise /app/data as a git repo so the built-in `Built-in: Git` MCP
# server has somewhere to commit to by default. Idempotent — `git init` is
# a no-op on an existing repo, and we set a default identity so the agent's
# commits don't fail with "please tell me who you are". This pairs with the
# HF Dataset persistence so the agent's repo state survives Space restarts.
if [ ! -d /app/data/.git ]; then
    mkdir -p /app/data
    chown "$PUID:$PGID" /app/data
    gosu "$PUID:$PGID" git -C /app/data init -q -b main 2>/dev/null || true
    gosu "$PUID:$PGID" git -C /app/data config user.email "agent@odysseus.local" 2>/dev/null || true
    gosu "$PUID:$PGID" git -C /app/data config user.name "Odysseus Agent" 2>/dev/null || true
fi

# Run first-time setup as the app user so data/ files get the right ownership.
# setup.py is idempotent — skips auth.json / .env if they already exist.
# || true so a setup failure never prevents the container from starting.
gosu "$PUID:$PGID" python /app/setup.py || true

# Pre-register the bundled local LLM as the default chat model so the
# user lands on it the first time they open the UI (no Settings round
# trip required). Idempotent: only fills empty default_model slots and
# only inserts the endpoint when it doesn't already exist. No-op when
# ODYSSEUS_LOCAL_LLM != 1.
gosu "$PUID:$PGID" python /app/scripts/local_llm/seed_default.py || true

# HF Spaces persistence: launch the periodic-push watcher in the background
# alongside the main app. Uses PERSIST_INTERVAL (default 300s).
if [ -n "$HF_TOKEN" ] && [ -n "$PERSIST_REPO_ID" ]; then
    echo "[entrypoint] starting persistence watcher (interval=${PERSIST_INTERVAL:-300}s)..."
    gosu "$PUID:$PGID" python /app/scripts/hf_persistence.py watch &
fi

# Bundled local LLM (HF Spaces / any CPU-only deploy): download a small
# GGUF on first run and serve it via llama-cpp-python's OpenAI-compatible
# HTTP server on 127.0.0.1:8001. Lets the Space answer queries without
# any external API key. Controlled by ODYSSEUS_LOCAL_LLM=1 — off by
# default so vanilla `docker compose up` behavior is unchanged.
#
# Runs as the non-root app user so the downloaded model file (and the HF
# cache dir under $MODELS_DIR) inherit ownership consistent with the rest
# of /app, and the persistence watcher above can include $MODELS_DIR in
# its snapshot if PERSIST_REPO_ID's dataset is set up for it.
if [ "${ODYSSEUS_LOCAL_LLM:-0}" = "1" ]; then
    echo "[entrypoint] starting bundled local LLM server (default: Qwen2.5-3B-Instruct GGUF)..."
    mkdir -p "${MODELS_DIR:-/app/models}"
    chown -R "$PUID:$PGID" "${MODELS_DIR:-/app/models}" 2>/dev/null || true
    gosu "$PUID:$PGID" /app/scripts/local_llm/serve.sh &
fi

# ASA router (docs/asa/README.md, stage 6): LiteLLM on 127.0.0.1:${ASA_ROUTER_PORT:-4000}
# fronting asa-deepseek (Colibri / ASA swarm at $ASA_DEEPSEEK_BASE) and asa-fast (the
# bundled local LLM above). Registering runs first and synchronously so the master key file
# exists before the proxy reads it. Off by default; enable with ASA_ROUTER=1.
if [ "${ASA_ROUTER:-0}" = "1" ]; then
    echo "[entrypoint] starting ASA router..."
    gosu "$PUID:$PGID" python /app/asa/router/seed_endpoint.py || true
    gosu "$PUID:$PGID" /app/asa/router/serve.sh &
fi

# Drop root and run the actual app. `gosu` is preferred over `su` /
# `sudo` because it cleans up the process tree (no extra shell layer)
# so signals (SIGTERM from `docker stop`) reach uvicorn directly.
exec gosu "$PUID:$PGID" "$@"
