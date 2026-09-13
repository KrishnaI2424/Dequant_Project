"""Phase 0 analytical predictions: bytes per decode step.

Produces the numbers Phase 0 is supposed to report before any kernel exists:
the three-way byte split, the bandwidth-bound latency floor, and the LM head's
share of decode traffic before and after quantizing the layers to INT4.

    python scripts/decode_bytes_report.py
"""

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dequant.byte_model import (
    LLAMA_3_2_1B_CONFIG,
    decode_bytes,
    predicted_latency_s,
    spec_from_hf_config,
)

# RTX 5060 Ti (device 0): 128-bit GDDR7 @ 28 Gbps. See hardware.md.
PEAK_BW_GB_S = 448.0

SEQ_LENS = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]

MODELS = [
    ("gpt2", "gpt2"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("Llama-3.2-1B-Instruct (unverified)", None),
]


def load_spec(repo):
    if repo is None:
        return spec_from_hf_config(LLAMA_3_2_1B_CONFIG, name="Llama-3.2-1B-Instruct")
    cfg = json.loads(
        urllib.request.urlopen(
            f"https://huggingface.co/{repo}/resolve/main/config.json", timeout=30
        ).read()
    )
    return spec_from_hf_config(cfg, name=repo)


def mb(x):
    """Decimal MB, matching the GB/s convention used for bandwidth."""
    return x / 1e6


def report(name, spec):
    print(f"\n{'=' * 72}\n{name}")
    print(f"  {spec.n_layers} layers, hidden {spec.hidden}, "
          f"kv_heads {spec.n_kv_heads}x{spec.head_dim}, vocab {spec.vocab:,}, "
          f"tied={spec.tied_embedding}")

    # --- fp16 baseline, three-way split -----------------------------------
    print(f"\n  fp16 decode-step bytes @ {PEAK_BW_GB_S:.0f} GB/s theoretical peak")
    print(f"  {'seq_len':>8}{'layers MB':>12}{'lm_head MB':>12}{'kv MB':>10}"
          f"{'act MB':>9}{'total MB':>11}{'floor ms':>10}{'tok/s':>8}")
    for n in SEQ_LENS:
        b = decode_bytes(spec, n)
        ms = predicted_latency_s(b["total"], PEAK_BW_GB_S) * 1e3
        print(f"  {n:>8}{mb(b['layer_weights']):>12.1f}{mb(b['lm_head']):>12.1f}"
              f"{mb(b['kv_cache']):>10.1f}{mb(b['activations']):>9.2f}"
              f"{mb(b['total']):>11.1f}{ms:>10.2f}{1e3 / ms:>8.0f}")

    # --- the Amdahl bound (OUTLINE section 5) ------------------------------
    n = 4096
    fp16 = decode_bytes(spec, n)
    int4 = decode_bytes(spec, n, weight_dtype="int4", lm_head_dtype="fp16")
    both = decode_bytes(spec, n, weight_dtype="int4", lm_head_dtype="int4")
    int8h = decode_bytes(spec, n, weight_dtype="int4", lm_head_dtype="int8")

    print(f"\n  LM head share of decode bytes @ seq_len={n} (g128 symmetric):")
    for label, b in (("fp16 everything", fp16),
                     ("INT4 layers, fp16 head", int4),
                     ("INT4 layers, INT8 head", int8h),
                     ("INT4 layers, INT4 head", both)):
        share = b["lm_head"] / b["total"] * 100
        speedup = fp16["total"] / b["total"]
        print(f"    {label:<24}{mb(b['total']):>9.1f} MB total"
              f"{share:>8.1f}% head{speedup:>7.2f}x vs fp16")


if __name__ == "__main__":
    for name, repo in MODELS:
        report(name, load_spec(repo))
    print()
