"""Patch Colibri's deepseek_v41.c: overlap the expert workers' round trips (lossless).

cluster_moe_batch (pinned 0a5504a1) serves the workers of a layer one after another:
write worker 0's request, read its reply, then worker 1, and so on. Over a relay each
round trip costs 60-220 ms, so a layer pays the SUM of its workers' round trips. In
A9 (run 42330145070) that sum was ~214 s of a 410 s generation.

This patch writes every worker's first request before reading any reply, then reads
the replies in worker order. Bytes sent, bytes received and the contrib rows written
are exactly the same, so the output does not change; the layer now pays roughly the
SLOWEST worker's round trip instead of all of them. No deadlock: a worker reads its
whole request before writing its reply. Extra chunks for a worker with more than
V41_CLUSTER_MAX_ITEMS experts stay request-then-reply.

Idempotent; exits non-zero if the pinned code is not found.

    python -m asa.swarm.patch_colibri_parallel /path/to/colibri/c/deepseek_v41.c
"""
from __future__ import annotations

import sys

MARK = "/* asa-cluster-parallel patch */"

OLD = 'static void cluster_moe_batch(Model *m, int layer, const float *xc, int topk,\n                              const int *uniq, const int *head, const int *next,\n                              const int *count, int n_uniq, const float *weights,\n                              float *contrib, float *gathered, float *down) {\n    int dim = m->c.dim;\n    for (int wi = 0; wi < g_cluster_n; wi++) {\n        ClusterWorker *w = &g_cluster_workers[wi];\n        int *mine = xmalloc((size_t)(n_uniq > 0 ? n_uniq : 1) * sizeof(int), "worker experts");\n        int total = 0;\n        for (int u = 0; u < n_uniq; u++)\n            if ((uniq[u] + layer) % g_cluster_n == wi) mine[total++] = u;\n        for (int at0 = 0; at0 < total; at0 += V41_CLUSTER_MAX_ITEMS) {\n            int n = total - at0 < V41_CLUSTER_MAX_ITEMS ? total - at0 : V41_CLUSTER_MAX_ITEMS;\n            uint32_t v; char magic[8];\n            if (cluster_io(w->fd, (void *)COLI_CLUSTER_MAGIC, 8, 1)) goto fail;\n            v = COLI_CLUSTER_VERSION; if (cluster_u32(w->fd, &v, 1)) goto fail;\n            v = (uint32_t)layer; if (cluster_u32(w->fd, &v, 1)) goto fail;\n            v = (uint32_t)dim; if (cluster_u32(w->fd, &v, 1)) goto fail;\n            v = (uint32_t)m->c.moe_inter; if (cluster_u32(w->fd, &v, 1)) goto fail;\n            v = (uint32_t)n; if (cluster_u32(w->fd, &v, 1)) goto fail;\n            for (int j = 0; j < n; j++) {\n                int u = mine[at0 + j], cnt = count[u], at = 0;\n                ehit_mark(m, layer, uniq[u]);   /* the dashboard\'s HITS, as expert_slots_at marks it */\n                for (int d = head[u]; d >= 0; d = next[d], at++)\n                    memcpy(gathered + (size_t)at * dim, xc + (size_t)(d / topk) * dim,\n                           (size_t)dim * sizeof(float));\n                v = (uint32_t)uniq[u]; if (cluster_u32(w->fd, &v, 1)) goto fail;\n                v = (uint32_t)cnt; if (cluster_u32(w->fd, &v, 1)) goto fail;\n                if (cluster_io(w->fd, gathered, (size_t)cnt * dim * sizeof(float), 1)) goto fail;\n            }\n            if (cluster_io(w->fd, magic, 8, 0) || memcmp(magic, COLI_CLUSTER_MAGIC, 8)) goto fail;\n            if (cluster_u32(w->fd, &v, 0) || v != COLI_CLUSTER_VERSION) goto fail;\n            if (cluster_u32(w->fd, &v, 0) || v != 0) goto fail;\n            if (cluster_u32(w->fd, &v, 0) || v != (uint32_t)n) goto fail;\n            for (int j = 0; j < n; j++) {\n                int u = mine[at0 + j], cnt = count[u], at = 0;\n                uint32_t eid, nr;\n                if (cluster_u32(w->fd, &eid, 0) || cluster_u32(w->fd, &nr, 0) ||\n                    eid != (uint32_t)uniq[u] || nr != (uint32_t)cnt) goto fail;\n                if (cluster_io(w->fd, down, (size_t)cnt * dim * sizeof(float), 0)) goto fail;\n                for (int d = head[u]; d >= 0; d = next[d], at++) {\n                    float wt = weights[d];\n                    float *dst = contrib + (size_t)d * dim;\n                    const float *src = down + (size_t)at * dim;\n                    for (int i = 0; i < dim; i++) dst[i] = wt * src[i];\n                }\n            }\n        }\n        free(mine);\n        continue;\nfail:\n        free(mine);\n        fprintf(stderr, "[CLUSTER] expert worker %s:%d failed during layer %d batch\\n",\n                w->host, w->port, layer);\n        exit(1);\n    }\n}\n\n'

NEW = '/* __MARK__ one request: the experts mine[0..n) of this worker, rows in chain order */\nstatic int asa_cluster_send(Model *m, ClusterWorker *w, int layer, const float *xc, int topk,\n                            const int *mine, int n, const int *head, const int *next,\n                            const int *count, const int *uniq, float *gathered, int dim) {\n    uint32_t v;\n    if (cluster_io(w->fd, (void *)COLI_CLUSTER_MAGIC, 8, 1)) return -1;\n    v = COLI_CLUSTER_VERSION; if (cluster_u32(w->fd, &v, 1)) return -1;\n    v = (uint32_t)layer; if (cluster_u32(w->fd, &v, 1)) return -1;\n    v = (uint32_t)dim; if (cluster_u32(w->fd, &v, 1)) return -1;\n    v = (uint32_t)m->c.moe_inter; if (cluster_u32(w->fd, &v, 1)) return -1;\n    v = (uint32_t)n; if (cluster_u32(w->fd, &v, 1)) return -1;\n    for (int j = 0; j < n; j++) {\n        int u = mine[j], cnt = count[u], at = 0;\n        ehit_mark(m, layer, uniq[u]);   /* the dashboard\'s HITS, as expert_slots_at marks it */\n        for (int d = head[u]; d >= 0; d = next[d], at++)\n            memcpy(gathered + (size_t)at * dim, xc + (size_t)(d / topk) * dim,\n                   (size_t)dim * sizeof(float));\n        v = (uint32_t)uniq[u]; if (cluster_u32(w->fd, &v, 1)) return -1;\n        v = (uint32_t)cnt; if (cluster_u32(w->fd, &v, 1)) return -1;\n        if (cluster_io(w->fd, gathered, (size_t)cnt * dim * sizeof(float), 1)) return -1;\n    }\n    return 0;\n}\n\n/* __MARK__ its reply: each expert\'s rows, weighted into contrib exactly as before */\nstatic int asa_cluster_recv(ClusterWorker *w, const int *mine, int n, const int *head,\n                            const int *next, const int *count, const int *uniq,\n                            const float *weights, float *contrib, float *down, int dim) {\n    uint32_t v; char magic[8];\n    if (cluster_io(w->fd, magic, 8, 0) || memcmp(magic, COLI_CLUSTER_MAGIC, 8)) return -1;\n    if (cluster_u32(w->fd, &v, 0) || v != COLI_CLUSTER_VERSION) return -1;\n    if (cluster_u32(w->fd, &v, 0) || v != 0) return -1;\n    if (cluster_u32(w->fd, &v, 0) || v != (uint32_t)n) return -1;\n    for (int j = 0; j < n; j++) {\n        int u = mine[j], cnt = count[u], at = 0;\n        uint32_t eid, nr;\n        if (cluster_u32(w->fd, &eid, 0) || cluster_u32(w->fd, &nr, 0) ||\n            eid != (uint32_t)uniq[u] || nr != (uint32_t)cnt) return -1;\n        if (cluster_io(w->fd, down, (size_t)cnt * dim * sizeof(float), 0)) return -1;\n        for (int d = head[u]; d >= 0; d = next[d], at++) {\n            float wt = weights[d];\n            float *dst = contrib + (size_t)d * dim;\n            const float *src = down + (size_t)at * dim;\n            for (int i = 0; i < dim; i++) dst[i] = wt * src[i];\n        }\n    }\n    return 0;\n}\n\nstatic void cluster_moe_batch(Model *m, int layer, const float *xc, int topk,\n                              const int *uniq, const int *head, const int *next,\n                              const int *count, int n_uniq, const float *weights,\n                              float *contrib, float *gathered, float *down) {\n    /* __MARK__ Every worker\'s first request is written before any reply is read, so\n     * the round trips of one layer overlap instead of adding up. Each worker still\n     * gets the same bytes and its reply lands in the same contrib rows, so the\n     * result is identical. A worker reads its whole request before it writes, so\n     * writing ahead cannot deadlock; later chunks (more than V41_CLUSTER_MAX_ITEMS\n     * experts on one worker) stay strictly request-then-reply. */\n    int dim = m->c.dim, nw = g_cluster_n;\n    int **mine = xmalloc((size_t)(nw > 0 ? nw : 1) * sizeof(int *), "worker expert lists");\n    int *total = xmalloc((size_t)(nw > 0 ? nw : 1) * sizeof(int), "worker expert counts");\n    int wi = 0;\n    for (wi = 0; wi < nw; wi++) {\n        mine[wi] = xmalloc((size_t)(n_uniq > 0 ? n_uniq : 1) * sizeof(int), "worker experts");\n        total[wi] = 0;\n        for (int u = 0; u < n_uniq; u++)\n            if ((uniq[u] + layer) % nw == wi) mine[wi][total[wi]++] = u;\n    }\n#define ASA_CHUNK(at0) (total[wi] - (at0) < V41_CLUSTER_MAX_ITEMS ? total[wi] - (at0) : V41_CLUSTER_MAX_ITEMS)\n    for (wi = 0; wi < nw; wi++)\n        if (total[wi] > 0 && asa_cluster_send(m, &g_cluster_workers[wi], layer, xc, topk, mine[wi],\n                                              ASA_CHUNK(0), head, next, count, uniq, gathered, dim)) goto fail;\n    for (wi = 0; wi < nw; wi++) {\n        for (int at0 = 0; at0 < total[wi]; at0 += V41_CLUSTER_MAX_ITEMS) {\n            int n = ASA_CHUNK(at0);\n            if (at0 > 0 && asa_cluster_send(m, &g_cluster_workers[wi], layer, xc, topk, mine[wi] + at0,\n                                            n, head, next, count, uniq, gathered, dim)) goto fail;\n            if (asa_cluster_recv(&g_cluster_workers[wi], mine[wi] + at0, n, head, next, count, uniq,\n                                 weights, contrib, down, dim)) goto fail;\n        }\n    }\n#undef ASA_CHUNK\n    for (int i = 0; i < nw; i++) free(mine[i]);\n    free(mine); free(total);\n    return;\nfail:\n    fprintf(stderr, "[CLUSTER] expert worker %s:%d failed during layer %d batch\\n",\n            g_cluster_workers[wi].host, g_cluster_workers[wi].port, layer);\n    exit(1);\n}\n\n'


def patch(src: str) -> str:
    if MARK in src:
        return src
    if src.count(OLD) != 1:
        raise SystemExit("patch_colibri_parallel: cluster_moe_batch is not the pinned version")
    return src.replace(OLD, NEW.replace("/* __MARK__", MARK + " /*"))


def main(argv=None) -> int:
    path = (argv or sys.argv[1:])[0]
    with open(path) as f:
        src = f.read()
    out = patch(src)
    if out != src:
        with open(path, "w") as f:
            f.write(out)
        print(f"[patch_colibri_parallel] patched {path}")
    else:
        print(f"[patch_colibri_parallel] already patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
