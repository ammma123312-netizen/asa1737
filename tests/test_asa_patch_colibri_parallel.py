"""The cluster overlap patch replaces exactly the pinned cluster_moe_batch, once.

Token-exactness was checked on the dsv41 tiny fixture with 3 local workers behind
40 ms proxies (Matching tokens 8/8 before and after; 6.4 s -> 3.9 s).
"""
import pytest

from asa.swarm import patch_colibri_parallel as P

SRC = "/* before */\n" + P.OLD + "typedef struct { int eid, nr; float *inputs; } ClusterRequestItem;\n"


def test_patch_once_and_idempotent():
    out = P.patch(SRC)
    assert P.patch(out) == out and out.count(P.MARK) >= 3
    assert "__MARK__" not in out
    # helpers come before their caller; every request goes out before the first reply is read
    assert out.index("static int asa_cluster_send(") < out.index("static void cluster_moe_batch(")
    body = out[out.index("static void cluster_moe_batch("):]
    first_send = body.index("asa_cluster_send(")
    first_recv = body.index("asa_cluster_recv(")
    assert first_send < first_recv
    assert out.endswith("ClusterRequestItem;\n")


def test_same_wire_and_arithmetic():
    out = P.patch(SRC)
    # the protocol fields and the weighting line are carried over verbatim
    for line in ("v = (uint32_t)m->c.moe_inter;", "ehit_mark(m, layer, uniq[u]);",
                 "for (int i = 0; i < dim; i++) dst[i] = wt * src[i];",
                 "if ((uniq[u] + layer) % nw == wi)"):
        assert line in out


def test_refuses_drift():
    with pytest.raises(SystemExit):
        P.patch(SRC.replace("V41_CLUSTER_MAX_ITEMS", "V41_ITEMS"))
