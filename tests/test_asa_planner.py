"""Tests for the ASA feasibility planner and project catalog."""
from pathlib import Path

import yaml

from asa import planner
from asa.planner import HARDWARE, MODELS, plan

CATALOG = Path(__file__).resolve().parent.parent / "asa" / "projects.yaml"


def test_700b_class_model_does_not_fit_free_github_runner():
    p = plan(HARDWARE["gha-free-public"], MODELS["glm-5.2"])
    assert p.mode == "impossible"
    assert not p.fits_disk
    # Freeing every byte on a free runner still is not 429 GB.
    p = plan(HARDWARE["gha-free-public"], MODELS["glm-5.2"], disk_gb=80)
    assert p.mode == "impossible"


def test_700b_class_model_streams_on_larger_runner_and_desktop():
    for hw in ("gha-larger-64core", "desktop-nvme-64gb", "workstation-128gb"):
        p = plan(HARDWARE[hw], MODELS["glm-5.2"])
        assert p.mode == "expert-streaming", hw
        assert 0.05 < p.est_tokens_per_s < 5, (hw, p.est_tokens_per_s)


def test_estimate_is_within_2x_of_colibri_published_numbers():
    # Colibrì README: GLM-5.2 ~0.8 tok/s on 64 GB + NVMe, 1.83 on 128 GB.
    assert 0.4 <= plan(HARDWARE["desktop-nvme-64gb"], MODELS["glm-5.2"]).est_tokens_per_s <= 1.6
    assert 0.9 <= plan(HARDWARE["workstation-128gb"], MODELS["glm-5.2"]).est_tokens_per_s <= 3.7


def test_estimate_matches_measured_run_in_4cpu_container():
    # Measured with Colibrì 2.1.0 (commit 0a5504a) on 4 vCPU / 15 GB RAM:
    # Qwen3-Coder-30B-A3B decoded 1.97-2.16 tok/s end-to-end.
    p = plan(HARDWARE["ccr-container"], MODELS["qwen3-coder-30b-a3b"])
    assert p.mode == "expert-streaming"
    assert 1.0 <= p.est_tokens_per_s <= 4.0, p.est_tokens_per_s


def test_smoke_model_fits_free_runner_after_disk_cleanup():
    p = plan(HARDWARE["gha-free-public"], MODELS["qwen3-coder-30b-a3b"], disk_gb=60)
    assert p.fits_disk and p.fits_ram_min
    assert p.mode == "expert-streaming"


def test_small_model_is_in_ram_and_fastest():
    hw = HARDWARE["gha-free-public"]
    small = plan(hw, MODELS["qwen2.5-3b-q4"])
    big = plan(hw, MODELS["qwen3-coder-30b-a3b"], disk_gb=60)
    assert small.mode == "in-ram"
    assert small.est_tokens_per_s > big.est_tokens_per_s


def test_dense_model_too_big_for_ram_uses_layer_streaming():
    p = plan(HARDWARE["desktop-nvme-32gb"], MODELS["llama-3.1-70b-q4"])
    assert p.mode == "layer-streaming"
    assert p.engine == "airllm"


def test_cli_runs(capsys):
    assert planner.main(["--list"]) == 0
    assert planner.main(["--hardware", "gha-free-public", "--model", "glm-5.2", "--json"]) == 0
    out = capsys.readouterr().out
    assert '"mode": "impossible"' in out


def test_catalog_has_at_least_20_unique_projects():
    data = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    projects = data["projects"]
    assert len(projects) >= 20
    repos = [p["repo"] for p in projects]
    assert len(repos) == len(set(repos))
    categories = set(data["categories"])
    for p in projects:
        assert p["category"] in categories, p["name"]
        assert p["repo"].count("/") == 1, p["repo"]
        assert p["solves"], p["name"]


# --- advisor's plan: 700B on 8 GB VRAM + 16 GB RAM + PCIe4 NVMe ------------

ADVISOR = HARDWARE["consultant-8gbvram"]


def test_dense_700b_is_unusable_at_every_precision_on_advisor_box():
    for mid in ("dense-700b-fp16", "dense-700b-fp8", "dense-700b-iq2"):
        p = plan(ADVISOR, MODELS[mid])
        assert p.mode == "layer-streaming", mid
        assert p.est_tokens_per_s < 0.1, (mid, p.est_tokens_per_s)


def test_speculative_decoding_multiplies_dense_speed_by_tokens_per_pass():
    base = plan(ADVISOR, MODELS["dense-700b-iq2"])
    spec = plan(ADVISOR, MODELS["dense-700b-iq2"], spec_rows=4, spec_accept=2.5)
    assert abs(spec.est_tokens_per_s / base.est_tokens_per_s - 2.5) < 0.1


def test_speculative_decoding_barely_helps_streamed_moe():
    # A verify pass over several rows reads the union of their experts.
    base = plan(ADVISOR, MODELS["glm-5.2"])
    spec = plan(ADVISOR, MODELS["glm-5.2"], spec_rows=4, spec_accept=2.5)
    assert spec.est_tokens_per_s / base.est_tokens_per_s < 1.2


def test_moe_beats_dense_700b_by_an_order_of_magnitude_on_advisor_box():
    moe = plan(ADVISOR, MODELS["deepseek-r1-1.58bit"])
    dense = plan(ADVISOR, MODELS["dense-700b-iq2"], spec_rows=7, spec_accept=4)
    assert moe.est_tokens_per_s > 10 * dense.est_tokens_per_s


def test_invalid_spec_settings_are_rejected():
    import pytest
    with pytest.raises(ValueError):
        plan(ADVISOR, MODELS["glm-5.2"], spec_rows=2, spec_accept=5)
