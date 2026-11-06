"""ASA feasibility planner.

Answers one question with arithmetic instead of hope:
"Can model M run on machine H, and roughly how fast?"

The estimate is a memory-bandwidth model, which is what bounds LLM decoding
on CPUs and on any machine that streams weights from disk:

  * Every generated token has to read the *active* weights once
    (all weights for a dense model, only the routed experts for a MoE).
  * Bytes already in RAM are read at memory bandwidth.
  * Bytes that do not fit in RAM are read from disk (Colibrì streams MoE
    experts from NVMe; AirLLM loads a dense model layer by layer).

The numbers are order-of-magnitude estimates, not benchmarks. They are good
enough to tell "~5 tokens/s" from "~1 token per minute", which is the
decision that matters before spending hours of CI time on a 400 GB download.

Usage:
    python -m asa.planner --hardware gha-free-public --model glm-5.2
    python -m asa.planner --hardware desktop-nvme-64gb --all-models
    python -m asa.planner --list
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass

GB = 1e9


@dataclass(frozen=True)
class Hardware:
    id: str
    name: str
    vcpus: int
    ram_gb: float
    disk_gb: float          # usable free disk for model weights
    mem_bw_gbs: float       # effective memory bandwidth for decoding (GB/s)
    disk_bw_gbs: float      # sustained random-ish read bandwidth (GB/s)
    cost: str
    notes: str = ""


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    total_b: float          # total parameters, billions
    active_b: float         # parameters read per token, billions (== total for dense)
    weights_gb: float       # on-disk size of the quantized weights
    ram_min_gb: float       # smallest RAM the engine is known to start with
    engine: str             # recommended engine for this model
    notes: str = ""

    @property
    def is_moe(self) -> bool:
        return self.active_b < self.total_b

    @property
    def bytes_per_param(self) -> float:
        return self.weights_gb * GB / (self.total_b * GB)


# Hardware numbers for GitHub-hosted runners follow GitHub's published runner
# specs (standard Linux runner for public repos: 4 vCPU / 16 GB RAM / 14 GB SSD;
# private repos: 2 vCPU / 7-8 GB). The bandwidth figures are rough estimates.
HARDWARE: dict[str, Hardware] = {h.id: h for h in (
    Hardware("gha-free-public", "GitHub Actions ubuntu-latest (public repo)",
             4, 16, 14, 15, 0.3, "free",
             "14 GB guaranteed; deleting preinstalled toolchains frees roughly "
             "50-60 GB more (see --disk-gb). 6 h job limit."),
    Hardware("gha-free-private", "GitHub Actions ubuntu-latest (private repo)",
             2, 7, 14, 10, 0.3, "free minutes quota", "2000 min/month on the free plan."),
    Hardware("gha-larger-64core", "GitHub Actions larger runner, 64 cores",
             64, 256, 2040, 120, 1.5, "paid (per minute)",
             "Team/Enterprise plans only. Big enough to hold GLM-5.2 on disk."),
    Hardware("hf-space-cpu-basic", "Hugging Face Space cpu-basic",
             2, 16, 50, 10, 0.3, "free", "Ephemeral disk; sleeps when idle."),
    Hardware("desktop-nvme-32gb", "Desktop, 32 GB RAM, PCIe4 NVMe",
             12, 32, 1000, 50, 5.0, "owned"),
    Hardware("desktop-nvme-64gb", "Desktop, 64 GB RAM, PCIe4 NVMe",
             16, 64, 2000, 60, 5.0, "owned",
             "Colibrì reports ~0.8 tok/s for GLM-5.2 on a similar box."),
    Hardware("workstation-128gb", "Workstation, 128 GB unified/DDR5",
             16, 128, 2000, 200, 6.0, "owned",
             "Colibrì reports 1.83 tok/s for GLM-5.2 on a 128 GB Ryzen AI Max+."),
)}

# Disk sizes for Colibrì models are taken from Colibrì's own setup catalog
# (c/setup_catalog.py). Others are standard GGUF quant sizes.
MODELS: dict[str, Model] = {m.id: m for m in (
    Model("glm-5.2", "GLM-5.2 (744B MoE, int4)", 744, 40, 429.3, 16, "colibri",
          "Colibrì int4-g64 + int8 MTP head."),
    Model("deepseek-r1-1.58bit", "DeepSeek-R1 671B, Unsloth dynamic 1.58-bit", 671, 37, 131, 20,
          "llama.cpp", "Unsloth UD-IQ1_S GGUF; llama.cpp mmaps what does not fit."),
    Model("llama-3.1-405b-q4", "Llama 3.1 405B dense, Q4_K_M", 405, 405, 243, 8, "airllm",
          "Dense: every token touches every weight. Layer streaming only."),
    Model("llama-3.1-70b-q4", "Llama 3.1 70B dense, Q4_K_M", 70, 70, 42.5, 8, "airllm"),
    Model("qwen3.6-35b-a3b", "Qwen3.6-35B-A3B (int4)", 35, 3, 23.1, 10, "colibri"),
    Model("qwen3-coder-30b-a3b", "Qwen3-Coder-30B-A3B (int4)", 30.5, 3.3, 19.4, 8, "colibri",
          "Prebuilt Colibrì container — the smoke-test model."),
    Model("olmoe-1b-7b", "OLMoE-1B-7B", 6.9, 1.3, 14, 8, "colibri"),
    Model("qwen2.5-3b-q4", "Qwen2.5-3B-Instruct Q4_K_M", 3.1, 3.1, 2.0, 4, "llama.cpp",
          "The model this repo already bundles for fast first answers."),
)}

RAM_OVERHEAD_GB = 3.0  # OS, runtime, KV cache for a short context


@dataclass
class Plan:
    hardware: str
    model: str
    fits_disk: bool
    fits_ram_min: bool
    mode: str                 # "in-ram", "expert-streaming", "layer-streaming", "impossible"
    resident_fraction: float  # share of per-token bytes served from RAM
    est_tokens_per_s: float
    est_seconds_per_token: float
    est_seconds_200_tokens: float
    verdict: str
    engine: str
    reasons: list

    def to_dict(self) -> dict:
        return asdict(self)


def _cache_hit(cache_gb: float, expert_gb: float) -> float:
    """Expert-cache hit rate. MoE routing is skewed (a few experts are hot),
    so hit rate grows much faster than the cached share. share**0.25 is a
    deliberately simple stand-in for that skew, calibrated so the planner
    lands within ~2x of Colibrì's published GLM-5.2 numbers (0.8 tok/s on a
    64 GB NVMe desktop, 1.83 tok/s on a 128 GB workstation)."""
    if expert_gb <= 0:
        return 1.0
    share = max(0.0, min(1.0, cache_gb / expert_gb))
    return min(0.98, share ** 0.25)


def plan(hw: Hardware, m: Model, disk_gb: float | None = None) -> Plan:
    disk = hw.disk_gb if disk_gb is None else disk_gb
    reasons: list[str] = []
    fits_disk = m.weights_gb <= disk
    fits_ram_min = m.ram_min_gb <= hw.ram_gb
    if not fits_disk:
        reasons.append(f"weights need {m.weights_gb:.0f} GB of disk, machine has {disk:.0f} GB")
    if not fits_ram_min:
        reasons.append(f"engine needs >= {m.ram_min_gb:.0f} GB RAM, machine has {hw.ram_gb:.0f} GB")

    bpp = m.bytes_per_param
    active_gb = m.active_b * bpp
    usable_ram = max(0.0, hw.ram_gb - RAM_OVERHEAD_GB)

    if m.weights_gb <= usable_ram:
        mode, resident = "in-ram", 1.0
    elif m.is_moe:
        # Dense share (attention, shared experts, embeddings) stays resident;
        # the rest of RAM is an expert cache. ~1/8 of active params are dense
        # in current large MoEs — rough but stable across GLM/DeepSeek/Qwen.
        dense_gb = min(usable_ram, max(active_gb / 8, m.weights_gb * 0.03))
        expert_total_gb = m.weights_gb - dense_gb
        hit = _cache_hit(usable_ram - dense_gb, expert_total_gb)
        dense_share = (active_gb / 8) / active_gb
        resident = dense_share + (1 - dense_share) * hit
        mode = "expert-streaming"
    else:
        resident = min(1.0, usable_ram / m.weights_gb)
        mode = "layer-streaming"

    ram_bytes = active_gb * resident
    disk_bytes = active_gb * (1 - resident)
    sec_per_tok = ram_bytes / hw.mem_bw_gbs + disk_bytes / hw.disk_bw_gbs
    tps = 1.0 / sec_per_tok if sec_per_tok > 0 else float("inf")

    engine = m.engine
    if mode == "in-ram" and m.engine == "airllm":
        engine = "llama.cpp"  # no need to stream if it fits

    if not (fits_disk and fits_ram_min):
        verdict, mode = "impossible", "impossible"
    elif tps >= 5:
        verdict = "interactive (good for chat)"
    elif tps >= 1:
        verdict = "usable but slow (chat with patience, fine for background jobs)"
    elif tps >= 0.1:
        verdict = "batch only (seconds per token; run as a background job)"
    else:
        verdict = "technically runs, practically unusable (minutes per answer line)"

    return Plan(
        hardware=hw.id, model=m.id, fits_disk=fits_disk, fits_ram_min=fits_ram_min,
        mode=mode, resident_fraction=round(resident, 3),
        est_tokens_per_s=round(tps, 3), est_seconds_per_token=round(sec_per_tok, 2),
        est_seconds_200_tokens=round(sec_per_tok * 200, 1),
        verdict=verdict, engine=engine, reasons=reasons,
    )


def _fmt(p: Plan) -> str:
    lines = [f"[{p.hardware}] {p.model}: {p.verdict}"]
    if p.mode != "impossible":
        lines.append(f"  mode={p.mode} engine={p.engine} resident={p.resident_fraction:.0%} "
                     f"~{p.est_tokens_per_s} tok/s (~{p.est_seconds_200_tokens:.0f}s for 200 tokens)")
    for r in p.reasons:
        lines.append(f"  - {r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="asa.planner", description=__doc__.split("\n\n")[0])
    ap.add_argument("--hardware", default="gha-free-public", choices=sorted(HARDWARE))
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--disk-gb", type=float, default=None,
                    help="override usable disk (e.g. after freeing space on a runner)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--list", action="store_true", help="list hardware and model presets")
    a = ap.parse_args(argv)

    if a.list:
        for h in HARDWARE.values():
            print(f"hw     {h.id:22} {h.vcpus:>3} vCPU {h.ram_gb:>5.0f} GB RAM {h.disk_gb:>6.0f} GB disk  {h.cost}")
        for m in MODELS.values():
            print(f"model  {m.id:22} {m.total_b:>6.1f}B total {m.active_b:>6.1f}B active {m.weights_gb:>7.1f} GB")
        return 0

    models = list(MODELS.values()) if a.all_models or not a.model else [MODELS[a.model]]
    plans = [plan(HARDWARE[a.hardware], m, a.disk_gb) for m in models]
    if a.json:
        print(json.dumps([p.to_dict() for p in plans], indent=2))
    else:
        print("\n".join(_fmt(p) for p in plans))
    return 0


if __name__ == "__main__":
    sys.exit(main())
