"""Patch Colibri's deepseek_v41.c for the ASA swarm before building it.

1. TCP_NODELAY on both ends of the expert-worker connection. The cluster protocol
   writes each request as many small fields and waits for the reply; without
   TCP_NODELAY, Nagle's algorithm plus delayed ACKs stall every layer's round trip.
   Measured locally on Colibri's tiny DSv4.1 fixture (same tokens either way):
   0.23 s with no cluster, 4.81 s through one localhost worker (~95 ms per round
   trip), see docs/asa/README.md. Pure transport change: no value moves.
2. V41_CLUSTER_MAX_WORKERS 16 -> 32, so a free-plan swarm can use 19 workers.

Idempotent; exits non-zero if the code it expects is not there (pinned commit drift).

    python -m asa.swarm.patch_colibri /path/to/colibri/c/deepseek_v41.c
"""
from __future__ import annotations

import sys

MARK = "/* asa-swarm patch */"

EDITS = [
    ("#include <netinet/in.h>\n",
     "#include <netinet/in.h>\n#include <netinet/tcp.h> " + MARK + "\n"),
    ("    out->fd = fd;\n",
     "    { int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one); } " + MARK + "\n"
     "    out->fd = fd;\n"),
    ("        int cfd = accept(fd, NULL, NULL);\n"
     "        if (cfd < 0) { if (errno == EINTR) continue; break; }\n",
     "        int cfd = accept(fd, NULL, NULL);\n"
     "        if (cfd < 0) { if (errno == EINTR) continue; break; }\n"
     "        { int one = 1; setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one); } " + MARK + "\n"),
    ("#define V41_CLUSTER_MAX_WORKERS 16\n",
     "#define V41_CLUSTER_MAX_WORKERS 32 " + MARK + "\n"),
]


def patch(src: str) -> str:
    if MARK in src:
        return src
    for old, new in EDITS:
        if src.count(old) != 1:
            raise SystemExit(f"patch_colibri: expected exactly one match for {old.strip()[:60]!r}")
        src = src.replace(old, new)
    return src


def main(argv=None) -> int:
    path = (argv or sys.argv[1:])[0]
    with open(path) as f:
        src = f.read()
    out = patch(src)
    if out != src:
        with open(path, "w") as f:
            f.write(out)
        print(f"[patch_colibri] patched {path}")
    else:
        print(f"[patch_colibri] already patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
