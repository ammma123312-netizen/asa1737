"""The Colibri patch applies exactly once and refuses a source it does not recognise."""
import pytest

from asa.swarm.patch_colibri import MARK, patch

SRC = ("#include <netinet/in.h>\n#define V41_CLUSTER_MAX_WORKERS 16\n"
       "    out->fd = fd;\n"
       "        int cfd = accept(fd, NULL, NULL);\n"
       "        if (cfd < 0) { if (errno == EINTR) continue; break; }\n")


def test_patch_adds_nodelay_both_ends_and_raises_limit():
    out = patch(SRC)
    assert out.count("TCP_NODELAY") == 2 and "#include <netinet/tcp.h>" in out
    assert "#define V41_CLUSTER_MAX_WORKERS 32" in out
    assert patch(out) == out and out.count(MARK) == 4


def test_patch_refuses_unknown_source():
    with pytest.raises(SystemExit):
        patch("int main(void) { return 0; }\n")
