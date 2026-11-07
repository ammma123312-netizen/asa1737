"""The engram prefetch patch applies once, in the right places, and refuses drifted source."""
import pytest

from asa.swarm.patch_colibri import patch as patch_nodelay
from asa.swarm.patch_colibri_engram import MARK, patch

# The fragments of deepseek_v41.c @ 0a5504a1 the patch anchors on (and the TCP patch's).
SRC = (
    "#include <netinet/in.h>\n#include <pthread.h>\n#define V41_CLUSTER_MAX_WORKERS 16\n"
    "static void engram_hash(const Engram *e, int table, int pos, int64_t *out) {\n}\n"
    "static const float *engram_row(EngramTable *t, int64_t id, int head_dim) {\n"
    "    int slot = (int)((uint64_t)(id * 0x9E3779B97F4A7C15ull) >> 40) & t->mask;\n"
    "    for (int probe = 0; probe < 4; probe++) {\n    }\n}\n"
    "/* model.py Engram.forward: look the n-grams up, project them into one key per residual\n"
    "static void engram_run(Model *m, int layer, float *h, int n, int start_pos) {\n"
    "    int table = l->engram_index, hc = c->hc_mult, dim = c->dim;\n"
    "    double started = now_s();\n"
    "            engram_hash(e, table, start_pos + t0 + r, ids);\n"
    "}\n"
    "static void forward_full(void) {\n"
    "    if (m->engram.active) engram_push(&m->engram, ids, n, image_mask);\n"
    "}\n"
    "static void forward_rows(void) {\n"
    "    for (int t = 0; t < n; t++) {\n"
    "        if (m->engram.active) engram_push(&m->engram, ids + t, 1, NULL);\n"
    "    }\n"
    "}\n"
    "    out->fd = fd;\n"
    "        int cfd = accept(fd, NULL, NULL);\n"
    "        if (cfd < 0) { if (errno == EINTR) continue; break; }\n"
)


def test_patch_places_prefetch_hooks():
    out = patch(SRC)
    assert patch(out) == out                      # idempotent
    # machinery defined before engram_run, which waits for its own table first
    assert out.index("static void engram_pf_launch(Model *m)") < out.index("static void engram_run(")
    run = out[out.index("static void engram_run("):]
    assert run.index("engram_pf_wait(table);") < run.index("engram_hash(e, table, start_pos + t0 + r")
    # forward_full: collect + launch right after the push
    full = out[out.index("static void forward_full"):out.index("static void forward_rows")]
    assert full.index("engram_push(") < full.index("engram_pf_collect(m, start_pos, n); engram_pf_launch(m);")
    # forward_rows: collect per bound slot inside the loop, launch once after it
    rows = out[out.index("static void forward_rows"):]
    assert rows.index("engram_pf_collect(m, pos[t], 1);") < rows.index("    }\n") < rows.index("engram_pf_launch(m);")
    assert out.count(MARK) == 5
    # the read-only cache probe matches engram_row's hashing and probe length
    assert "& t->mask;\n    for (int probe = 0; probe < 4; probe++)\n        if (t->key" in out
    # prefetch can be switched off, and never runs in dry mode
    assert 'getenv("V41_ENGRAM_PREFETCH")' in out and "!g_v41_dry" in out


def test_composes_with_nodelay_patch_in_either_order():
    a, b = patch_nodelay(patch(SRC)), patch(patch_nodelay(SRC))
    assert a == b and "TCP_NODELAY" in a and "engram_pf_wait" in a


@pytest.mark.parametrize("drop", [
    "#include <pthread.h>\n",
    "    if (m->engram.active) engram_push(&m->engram, ids, n, image_mask);\n",
    "    int slot = (int)((uint64_t)(id * 0x9E3779B97F4A7C15ull) >> 40) & t->mask;\n",
])
def test_refuses_drifted_source(drop):
    with pytest.raises(SystemExit):
        patch(SRC.replace(drop, ""))
