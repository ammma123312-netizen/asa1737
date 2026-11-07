"""Patch Colibri's deepseek_v41.c: prefetch a step's engram rows before layer 1 (lossless).

The two engram tables (layers 1 and 14) are read one row at a time inside engram_run:
~48 rows per table per token, each a pread of 256 + 8 bytes. On an HFFS (FUSE) mount every
miss is a 60-80 ms network round trip, and they run one after another.

After engram_push records the step's tokens, the n-gram row ids of every position in the step
are already determined (engram_hash depends only on the history). This patch computes them for
both tables right there, skips rows the engram cache already holds (a read-only lookup that
does not touch the LRU clock or counters), and reads the rest on background threads into a
scratch buffer that is thrown away. The engine's own reads in engram_run are unchanged; they
just find the bytes in the page cache / HFFS block cache instead of the network. Before using
a table, engram_run waits for that table's batch, so layer 1 waits only for its own rows while
layer 14's keep arriving during layers 1-13.

No value changes: the prefetch never writes engine state other than its own batch bookkeeping,
and the rows are still read, dequantized and cached by the unmodified engram_row.

Environment (read once):  V41_ENGRAM_PREFETCH=0 disables it;
                          V41_ENGRAM_PREFETCH_THREADS (default 16, max 64).

Idempotent; exits non-zero if the code it expects is not there (pinned commit drift).
Pinned against JustVugg/colibri 0a5504a1dc64f86fcd0fe7db298b0448c730e7e6.

    python -m asa.swarm.patch_colibri_engram /path/to/colibri/c/deepseek_v41.c
"""
from __future__ import annotations

import sys

MARK = "/* asa-engram-prefetch patch */"

# Inserted right before engram_run: the batch type, the reader threads, collect/launch/wait.
PREFETCH_C = r"""
/* ---------------------------------------------------- engram prefetch ---- */ __MARK__
/* Row ids of a step are known as soon as engram_push has run, so read them concurrently
 * ahead of engram_run instead of one network round trip at a time. Read-only and
 * discarding: engram_row still reads, dequantizes and caches every row itself. */
typedef struct {
    EngramTable *t;
    int head_dim;
    int64_t *ids;
    int n, cap;
    pthread_t th[64];
    int threads, running;
    int next;                   /* shared work cursor */
    pthread_mutex_t lock;
} EngramPrefetch;

static EngramPrefetch g_eng_pf[V41_MAX_ENGRAM];
static int g_eng_pf_on = -1, g_eng_pf_threads = 16;

static int engram_pf_enabled(void) {
    if (g_eng_pf_on < 0) {
        const char *v = getenv("V41_ENGRAM_PREFETCH");
        g_eng_pf_on = !(v && v[0] == '0');
        const char *k = getenv("V41_ENGRAM_PREFETCH_THREADS");
        if (k && atoi(k) > 0) g_eng_pf_threads = atoi(k) > 64 ? 64 : atoi(k);
        for (int i = 0; i < V41_MAX_ENGRAM; i++) pthread_mutex_init(&g_eng_pf[i].lock, NULL);
    }
    return g_eng_pf_on && !g_v41_dry;
}

/* The same probe as engram_row, without touching the clock, counters or slots. */
static int engram_cached(const EngramTable *t, int64_t id) {
    int slot = (int)((uint64_t)(id * 0x9E3779B97F4A7C15ull) >> 40) & t->mask;
    for (int probe = 0; probe < 4; probe++)
        if (t->key[(slot + probe) & t->mask] == id) return 1;
    return 0;
}

static void engram_pf_read(const EngramTable *t, int64_t id, int head_dim, uint8_t *buf) {
    int groups = head_dim / 32;
    if (id < 0 || id >= t->rows) return;          /* engram_row reports it */
    if (t->gt >= 0) {
        if (pread(t->fd_w, buf, (size_t)t->rb, t->off_w + id * t->rb) < 0) return;
    } else if (t->inline_scales) {
        if (pread(t->fd_w, buf, (size_t)(head_dim + groups),
                  t->off_w + id * (head_dim + groups)) < 0) return;
    } else {
        if (pread(t->fd_w, buf, (size_t)head_dim, t->off_w + id * head_dim) < 0) return;
        if (pread(t->fd_s, buf, (size_t)groups, t->off_s + id * groups) < 0) return;
    }
}

static void *engram_pf_worker(void *arg) {
    EngramPrefetch *p = arg;
    uint8_t buf[4096];
    for (;;) {
        pthread_mutex_lock(&p->lock);
        int i = p->next++;
        pthread_mutex_unlock(&p->lock);
        if (i >= p->n) return NULL;
        engram_pf_read(p->t, p->ids[i], p->head_dim, buf);
    }
}

/* Wait for a table's outstanding batch (no-op when none is running). */
static void engram_pf_wait(int table) {
    EngramPrefetch *p = &g_eng_pf[table];
    if (!p->running) return;
    for (int i = 0; i < p->threads; i++) pthread_join(p->th[i], NULL);
    p->running = 0;
    p->n = 0;
}

/* Add the uncached row ids of positions [pos, pos + n) for every table. Call while the
 * slot whose history these positions belong to is bound. */
static void engram_pf_collect(Model *m, int pos, int n) {
    Engram *e = &m->engram;
    if (!e->active || !engram_pf_enabled()) return;
    int64_t ids[V41_MAX_NGRAM * V41_MAX_EHEADS];
    for (int table = 0; table < e->n_layers; table++) {
        EngramPrefetch *p = &g_eng_pf[table];
        engram_pf_wait(table);                 /* never grow a list a thread is reading */
        EngramTable *t = &e->table[table];
        if (t->rb > 4096 || e->head_dim + e->head_dim / 32 > 4096) continue;
        for (int r = 0; r < n; r++) {
            engram_hash(e, table, pos + r, ids);
            for (int col = 0; col < e->cols; col++) {
                if (engram_cached(t, ids[col])) continue;
                if (p->n == p->cap) {
                    int cap = p->cap ? p->cap * 2 : 256;
                    int64_t *grown = realloc(p->ids, (size_t)cap * sizeof(int64_t));
                    if (!grown) return;        /* prefetch is optional */
                    p->ids = grown; p->cap = cap;
                }
                p->ids[p->n++] = ids[col];
            }
        }
    }
}

/* Start the reader threads for every table that collected rows. */
static void engram_pf_launch(Model *m) {
    Engram *e = &m->engram;
    if (!e->active || !engram_pf_enabled()) return;
    for (int table = 0; table < e->n_layers; table++) {
        EngramPrefetch *p = &g_eng_pf[table];
        if (p->running || p->n == 0) continue;
        p->t = &e->table[table];
        p->head_dim = e->head_dim;
        p->next = 0;
        int want = p->n < g_eng_pf_threads ? p->n : g_eng_pf_threads;
        p->threads = 0;
        for (int i = 0; i < want; i++)
            if (pthread_create(&p->th[i], NULL, engram_pf_worker, p) == 0) p->threads++;
        p->running = p->threads > 0;
        if (!p->running) p->n = 0;
    }
}

"""

EDITS = [
    # 1. the prefetch machinery, right before engram_run
    ("/* model.py Engram.forward: look the n-grams up, project them into one key per residual\n",
     PREFETCH_C.replace("__MARK__", MARK).lstrip("\n")
     + "/* model.py Engram.forward: look the n-grams up, project them into one key per residual\n"),
    # 2. engram_run waits for its own table's batch before reading rows
    ("    int table = l->engram_index, hc = c->hc_mult, dim = c->dim;\n"
     "    double started = now_s();\n",
     "    int table = l->engram_index, hc = c->hc_mult, dim = c->dim;\n"
     "    double started = now_s();\n"
     "    engram_pf_wait(table); " + MARK + "\n"),
    # 3. forward_full: collect and launch right after the step's tokens enter the history
    ("    if (m->engram.active) engram_push(&m->engram, ids, n, image_mask);\n",
     "    if (m->engram.active) engram_push(&m->engram, ids, n, image_mask);\n"
     "    if (m->engram.active) { engram_pf_collect(m, start_pos, n); engram_pf_launch(m); } "
     + MARK + "\n"),
    # 4. forward_rows: one position per slot, collected while that slot is bound
    ("        if (m->engram.active) engram_push(&m->engram, ids + t, 1, NULL);\n"
     "    }\n",
     "        if (m->engram.active) engram_push(&m->engram, ids + t, 1, NULL);\n"
     "        if (m->engram.active) engram_pf_collect(m, pos[t], 1); " + MARK + "\n"
     "    }\n"
     "    if (m->engram.active) engram_pf_launch(m); " + MARK + "\n"),
]

# The code the edits rely on, checked so a drifted source fails loudly instead of compiling
# a patch whose assumptions no longer hold.
REQUIRED = [
    "static void engram_hash(const Engram *e, int table, int pos, int64_t *out) {",
    "static const float *engram_row(EngramTable *t, int64_t id, int head_dim) {",
    "int slot = (int)((uint64_t)(id * 0x9E3779B97F4A7C15ull) >> 40) & t->mask;",
    "for (int probe = 0; probe < 4; probe++) {",
    "#include <pthread.h>",
    "            engram_hash(e, table, start_pos + t0 + r, ids);\n",
]


def patch(src: str) -> str:
    if MARK in src:
        return src
    for needle in REQUIRED:
        if needle not in src:
            raise SystemExit(f"patch_colibri_engram: source lacks {needle.strip()[:70]!r}")
    for old, new in EDITS:
        if src.count(old) != 1:
            raise SystemExit(f"patch_colibri_engram: expected exactly one match for {old.strip()[:60]!r}")
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
        print(f"[patch_colibri_engram] patched {path}")
    else:
        print(f"[patch_colibri_engram] already patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
