#!/usr/bin/env bash
# Prepare one ASA swarm node (coordinator or worker) on a fresh Linux machine.
# Zero cost: weights are streamed from the public Hugging Face repo through HFFS
# (asa/swarm/hffs.py); nothing needs an account or key.
#
#   REPO=deepseek-ai/DeepSeek-V4.1-Flash ./asa/swarm/node_setup.sh
# Exports (via $GITHUB_ENV when present, else prints): VIEW, COLI
set -euo pipefail

REPO="${REPO:-deepseek-ai/DeepSeek-V4.1-Flash}"
WORK="${WORK:-$HOME/asa-swarm}"
COLIBRI_REF="${COLIBRI_REF:-0a5504a1dc64f86fcd0fe7db298b0448c730e7e6}"
CACHE_GB="${CACHE_GB:-40}"
# Coordinator: HFFS_ARGS="--warm-include . --warm-exclude '\.experts\.|engram'" prefetches the
# ~11 GB of dense tensors in parallel (measured 142 MB/s vs ~8 MB/s for the engine's own
# sequential load). Readahead stays off: dense and expert tensors are interleaved in
# every shard, so guessing the next bytes mostly fetched experts nobody asked for.
ENGINE="${ENGINE:-deepseek_v41}"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
SUDO=""; [ "$(id -u)" = "0" ] || SUDO="sudo"

mkdir -p "$WORK"
echo "[node] packages"
$SUDO apt-get update -qq >/dev/null
$SUDO apt-get install -y -qq fuse3 libfuse2t64 build-essential python3-venv >/dev/null 2>&1 \
  || $SUDO apt-get install -y -qq fuse3 libfuse2 build-essential python3-venv >/dev/null
grep -q '^user_allow_other' /etc/fuse.conf 2>/dev/null || echo user_allow_other | $SUDO tee -a /etc/fuse.conf >/dev/null

echo "[node] python env"
[ -x "$WORK/venv/bin/python" ] || python3 -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install -q fusepy requests transformers tokenizers numpy sympy

mount_hffs() {
    echo "[node] mount $REPO (HFFS, cache ${CACHE_GB} GB)"
    mkdir -p "$WORK/mnt" "$WORK/cache"
    (cd "$HERE" && nohup "$WORK/venv/bin/python" -m asa.swarm.hffs "$REPO" "$WORK/mnt" \
        --cache "$WORK/cache" --cache-gb "$CACHE_GB" --small-block '*-0004[78]-of-*' --small-block-kb 4 --readahead 0 \
        ${HFFS_ARGS:-} > "$WORK/hffs.log" 2>&1 &)
    for _ in $(seq 1 60); do [ -f "$WORK/mnt/config.json" ] && break; sleep 2; done
    [ -f "$WORK/mnt/config.json" ] || { cat "$WORK/hffs.log"; exit 1; }
    head -1 "$WORK/hffs.log"
}

if [ -n "${ASA_STAGE:-}" ]; then
    # Sparse local staging: this node's tensors at their real offsets, holes elsewhere.
    # The engine then reads local disk; FUSE stays only for files linked from the mount
    # (the coordinator's engram tables). ASA_MOUNT=1 mounts HFFS first for those links.
    [ "${ASA_MOUNT:-0}" = "1" ] && mount_hffs
    echo "[node] stage: $ASA_STAGE"
    set -f   # the stage arguments carry globs/regexes; do not let the shell expand them
    (cd "$HERE" && "$WORK/venv/bin/python" -m asa.swarm.stage "$REPO" "$WORK/view" $ASA_STAGE) | tee "$WORK/stage.log"
    set +f
else
    mount_hffs
    echo "[node] writable view"
    "$WORK/venv/bin/python" -c "import sys; sys.path.insert(0, '$HERE'); from asa.swarm.hffs import make_view; print(make_view('$WORK/mnt', '$WORK/view'), 'files linked')"
fi

echo "[node] Colibrì $ENGINE"
[ -d "$WORK/colibri/.git" ] || git clone -q https://github.com/JustVugg/colibri "$WORK/colibri"
git -C "$WORK/colibri" checkout -q "$COLIBRI_REF"
# TCP_NODELAY on the cluster sockets + 32-worker limit (see asa/swarm/patch_colibri.py)
(cd "$HERE" && python3 -m asa.swarm.patch_colibri "$WORK/colibri/c/deepseek_v41.c")
# engram rows of a step read concurrently ahead of layer 1 (token-exact; see patch_colibri_engram.py)
(cd "$HERE" && python3 -m asa.swarm.patch_colibri_engram "$WORK/colibri/c/deepseek_v41.c")
# all workers of a layer queried before any reply is read (token-exact; see patch_colibri_parallel.py)
(cd "$HERE" && python3 -m asa.swarm.patch_colibri_parallel "$WORK/colibri/c/deepseek_v41.c")
(cd "$WORK/colibri/c" && ./setup.sh >/dev/null && make -s "$ENGINE")
if [ "$ENGINE" = "deepseek_v41" ]; then
    (cd "$WORK/colibri" && "$WORK/venv/bin/python" c/tools/prepare_dsv41.py --model "$WORK/view" | tail -1)
fi

{
  echo "VIEW=$WORK/view"
  echo "COLI=$WORK/colibri/c/coli"
  echo "ASA_WORK=$WORK"
} | if [ -n "${GITHUB_ENV:-}" ]; then tee -a "$GITHUB_ENV"; else cat; fi
echo "[node] ready"
