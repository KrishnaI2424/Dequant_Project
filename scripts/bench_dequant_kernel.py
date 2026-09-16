"""How fast is the dequant-GEMV kernel, per projection shape and per format?

scripts/bandwidth_by_dtype.py already proved the storage claim in isolation:
reading INT4-packed bytes is 2.91x faster than reading the same weights as
fp16. That script moved bytes and did nothing with them. This one runs the
real kernel -- unpack in registers, multiply by the activation, reduce --
on Llama-3.2-3B's four projection shapes, and asks whether the arithmetic
is free, i.e. whether the kernel still lands near the DRAM read ceiling.

Why L2 must be defeated
-----------------------
The RTX 5060 Ti has a 32 MB L2. A q_proj weight at INT4 g128 is 4.9 MB, so a
naive loop that hammers ONE weight tensor measures L2 bandwidth, not DRAM
bandwidth, and reports a number several times too good. Every (shape, format)
pair here therefore allocates `n_copies = max(28, ceil(8 * L2 / bytes))`
DISTINCT weights and walks all of them back to back inside one captured CUDA
graph, so by the time a weight comes round again it has long been evicted.
28 is the floor because that is how many layers the model has -- a per-kernel
time measured over fewer copies than the model itself rotates through would
be optimistic for the wrong reason.

One CUDA graph, not a Python loop: at these sizes a single GEMV is a few tens
of microseconds, which is the same order as HuggingFace-free eager dispatch,
so an untimed launch gap would be folded into the kernel time. Graph replay
costs almost no CPU (same argument as dequant/bench.py's "graph" arm).

The %-of-peak column uses 421.8 GB/s, the measured pure-READ ceiling from
hardware.md section 2b (fp16 .sum over a 1 GiB buffer, 94.2% of the 448 GB/s
theoretical). NOT the 385.4 GB/s copy figure -- a GEMV reads weights and
writes only a tiny output vector, so the read ceiling is the honest
denominator.

    python scripts/bench_dequant_kernel.py
    python scripts/bench_dequant_kernel.py --profile     # ncu target, eager
    python scripts/bench_dequant_kernel.py --e2e         # whole-model decode
"""

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dequant.bench import summarize, time_decode_step
from dequant.byte_model import decode_bytes, spec_from_hf_config
from dequant.packer import QuantConfig, quantize_tensor
from dequant.quant_linear import gemv, quantize_model

# Measured pure-read ceiling on this card; hardware.md section 2b.
READ_CEILING_GB_S = 421.8

# Llama-3.2-3B projections as (label, N, K). q/o and k/v share a shape, as do
# gate and up, so four rows cover all seven matrices.
SHAPES = [
    ("q/o_proj", 3072, 3072),
    ("k/v_proj", 1024, 3072),
    ("gate/up_proj", 8192, 3072),
    ("down_proj", 3072, 8192),
]

# One axis varied at a time off the INT4 g128 baseline, plus fp16 as the
# reference every speedup is measured against.
FORMATS = [
    ("fp16 F.linear", None),
    ("INT8 g128 sym", QuantConfig(8, 128, True)),
    ("INT4 g128 sym", QuantConfig(4, 128, True)),
    ("INT4 g128 asym", QuantConfig(4, 128, False)),
    ("INT4 g32 sym", QuantConfig(4, 32, True)),
    ("INT4 per-chan sym", QuantConfig(4, None, True)),
    ("INT4 g128 sym inter", QuantConfig(4, 128, True, "interleaved")),
]

M = 1                       # decode batch: one token
MIN_COPIES = 28             # one per decoder layer
L2_MULTIPLE = 8             # working set must be this many times the L2
FALLBACK_L2_BYTES = 32 * 2 ** 20

MODEL = "unsloth/Llama-3.2-3B-Instruct"
PROMPT = "The capital of France is"
E2E_CFG = QuantConfig(4, 128, False)   # best accuracy format, per eval_quantized
E2E_SEQ_LEN = 512
FP16_REFERENCE_TOK_S = 52.0            # PROGRESS.md Phase 0, seq 512, graph
PHASE1_TARGET_TOK_S = 148.0


def l2_bytes():
    props = torch.cuda.get_device_properties(0)
    return getattr(props, "L2_cache_size", None) or FALLBACK_L2_BYTES


def one_copy(n, k, cfg):
    """One distinct weight, packed if cfg is given, else plain fp16."""
    w = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.02
    return w if cfg is None else quantize_tensor(w, cfg)


def capture(fn):
    """Capture fn() into a CUDA graph, with the side-stream warmup."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def time_pair(n, k, cfg, warmup, iters):
    """Median per-kernel time for one (shape, format) pair. Frees as it goes."""
    first = one_copy(n, k, cfg)
    bytes_per_copy = n * k * 2 if cfg is None else first.nbytes
    n_copies = max(MIN_COPIES,
                   math.ceil(L2_MULTIPLE * l2_bytes() / bytes_per_copy))

    weights = [first] + [one_copy(n, k, cfg) for _ in range(n_copies - 1)]
    x = torch.randn(M, k, dtype=torch.float16, device="cuda")
    ys = [None] * n_copies

    if cfg is None:
        def step():
            for i, w in enumerate(weights):
                ys[i] = F.linear(x, w)
    else:
        def step():
            for i, pw in enumerate(weights):
                ys[i] = gemv(x, pw)

    graph = capture(step)

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start_ev.record()
        graph.replay()
        end_ev.record()
        torch.cuda.synchronize()
        samples.append(start_ev.elapsed_time(end_ev))

    stats = summarize(samples)
    per_call_bytes = bytes_per_copy + x.nbytes + M * n * 2

    del graph, weights, ys, x, step, first
    torch.cuda.empty_cache()

    return {
        "n_copies": n_copies,
        "bytes_per_call": per_call_bytes,
        "replay_ms": stats,
        "us_median": stats["median"] * 1e3 / n_copies,
        "us_p25": stats["p25"] * 1e3 / n_copies,
        "us_p75": stats["p75"] * 1e3 / n_copies,
    }


def microbench(args, record):
    print(f"  M={M}, one captured graph of n_copies back-to-back calls, "
          f"read ceiling {READ_CEILING_GB_S:.1f} GB/s (hardware.md 2b)\n")
    print(f"  {'shape':<14}{'format':<20}{'N':>6}{'K':>6}{'cop':>5}"
          f"{'KB/call':>9}{'us':>9}{'p25-p75':>16}{'GB/s':>8}{'%ceil':>7}"
          f"{'floor us':>10}{'vs fp16':>9}")

    for label, n, k in SHAPES:
        if args.shapes and not any(s in label.lower() for s in args.shapes):
            continue
        base_us = None
        for fmt_label, cfg in FORMATS:
            t = time_pair(n, k, cfg, args.warmup, args.iters)
            b = t["bytes_per_call"]
            us = t["us_median"]
            gb_s = b / us * 1e-3                 # bytes / (us * 1e-6) / 1e9
            floor_us = b / (READ_CEILING_GB_S * 1e9) * 1e6
            if base_us is None:
                base_us = us
            row = {
                "shape": label, "format": fmt_label, "N": n, "K": k,
                "gb_s": gb_s, "pct_ceiling": gb_s / READ_CEILING_GB_S * 100,
                "floor_us": floor_us, "speedup_vs_fp16": base_us / us,
                **t,
            }
            record["rows"].append(row)
            print(f"  {label:<14}{fmt_label:<20}{n:>6}{k:>6}{t['n_copies']:>5}"
                  f"{b / 1e3:>9.1f}{us:>9.2f}"
                  f"{t['us_p25']:>8.2f}-{t['us_p75']:<7.2f}{gb_s:>8.1f}"
                  f"{gb_s / READ_CEILING_GB_S * 100:>6.1f}%{floor_us:>10.2f}"
                  f"{base_us / us:>8.2f}x")
        print()


def profile(args):
    """Eager launches for ncu. No graph, no timing -- just a clean target.

    ncu's dram__bytes_read.sum for the single profiled launch should land
    within a few percent of the byte count printed here; anything much lower
    means the weight is being served from L2, anything higher means the
    kernel is re-reading something it should not.
    """
    print(f"eager profile target: INT4 g128 asym, M={M}\n")
    for label, n, k in SHAPES:
        if args.shapes and not any(s in label.lower() for s in args.shapes):
            continue
        pw = quantize_tensor(
            torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.02,
            QuantConfig(4, 128, False))
        x = torch.randn(M, k, dtype=torch.float16, device="cuda")
        expected = pw.nbytes + x.nbytes
        print(f"  {label:<14} [{n},{k}]  expected DRAM read "
              f"{expected:,} B ({expected / 1e6:.2f} MB)")
        for _ in range(5):
            gemv(x, pw)
        torch.cuda.synchronize()
        gemv(x, pw)                # <- the launch to profile (--launch-skip 5)
        del pw, x
        torch.cuda.empty_cache()


def e2e(args, record):
    """Whole-model decode: does the kernel actually make the model faster?"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    model = model.to("cuda").eval()

    # Captured before quantization: model.config is unchanged by the swap, but
    # taking it first keeps the byte model's input unambiguous.
    spec = spec_from_hf_config(model.config.to_dict(), name=MODEL)
    prompt_ids = tok(PROMPT, return_tensors="pt").to("cuda")

    def generate():
        with torch.no_grad():
            out = model.generate(**prompt_ids, max_new_tokens=8, do_sample=False)
        return tok.decode(out[0][prompt_ids.input_ids.shape[1]:]).strip()

    print(f"{MODEL}  seq_len={E2E_SEQ_LEN}, CUDA-graph capture\n")

    fp16_text = generate()
    fp16 = time_decode_step(model, E2E_SEQ_LEN, cache_impl="graph",
                            warmup=10, iters=30)
    fp16_ms = fp16["total_ms"]["median"]
    fp16_tok_s = 1e3 / fp16_ms
    print(f"  fp16   {fp16_ms:7.3f} ms  {fp16_tok_s:6.1f} tok/s   {fp16_text!r}")

    swap = quantize_model(model, E2E_CFG)
    print(f"\n  quantize_model({E2E_CFG}) -> {swap}")
    print(f"  weights {swap['fp16_bytes'] / 1e6:.0f} -> "
          f"{swap['packed_bytes'] / 1e6:.0f} MB "
          f"({swap['fp16_bytes'] / swap['packed_bytes']:.2f}x)\n")

    int4_text = generate()
    int4 = time_decode_step(model, E2E_SEQ_LEN, cache_impl="graph",
                            warmup=10, iters=30)
    int4_ms = int4["total_ms"]["median"]
    int4_tok_s = 1e3 / int4_ms
    print(f"  INT4   {int4_ms:7.3f} ms  {int4_tok_s:6.1f} tok/s   {int4_text!r}")

    # byte_model._quant_bytes DOES include the fp16 scale (and zero-point when
    # asymmetric) per group, so no manual packed-overhead correction is needed
    # here; the group_size/asymmetric arguments must match E2E_CFG.
    predicted = decode_bytes(spec, E2E_SEQ_LEN, weight_dtype="int4",
                             lm_head_dtype="fp16", group_size=E2E_CFG.group_size,
                             asymmetric=not E2E_CFG.symmetric)["total"]
    achieved = predicted / (int4_ms * 1e-3) / 1e9

    print(f"\n  speedup {int4_tok_s / fp16_tok_s:.2f}x  "
          f"(vs {FP16_REFERENCE_TOK_S:.0f} tok/s reference: "
          f"{int4_tok_s / FP16_REFERENCE_TOK_S:.2f}x, "
          f"vs {PHASE1_TARGET_TOK_S:.0f} tok/s Phase 1 target: "
          f"{int4_tok_s / PHASE1_TARGET_TOK_S * 100:.0f}%)")
    print(f"  INT4 predicted bytes {predicted / 1e6:.1f} MB -> "
          f"{achieved:.1f} GB/s ({achieved / READ_CEILING_GB_S * 100:.1f}% of ceiling)")

    record["e2e"] = {
        "model": MODEL, "seq_len": E2E_SEQ_LEN,
        "format": str(E2E_CFG), "swap": swap,
        "fp16_ms": fp16_ms, "fp16_tok_s": fp16_tok_s, "fp16_text": fp16_text,
        "int4_ms": int4_ms, "int4_tok_s": int4_tok_s, "int4_text": int4_text,
        "speedup": int4_tok_s / fp16_tok_s,
        "predicted_bytes": predicted, "achieved_gb_s": achieved,
        "pct_ceiling": achieved / READ_CEILING_GB_S * 100,
        "fp16_reference_tok_s": FP16_REFERENCE_TOK_S,
        "phase1_target_tok_s": PHASE1_TARGET_TOK_S,
        "fp16_decode": fp16, "int4_decode": int4,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--profile", action="store_true",
                    help="eager INT4 g128 asym launches for ncu; no timing")
    ap.add_argument("--e2e", action="store_true",
                    help="also run the whole-model decode comparison")
    ap.add_argument("--shapes", default="",
                    help="comma-separated substrings of the shape labels")
    args = ap.parse_args()
    args.shapes = [s.strip().lower() for s in args.shapes.split(",") if s.strip()]

    if args.profile:
        profile(args)
        return

    props = torch.cuda.get_device_properties(0)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "gpu": props.name,
            "sm": f"{props.major}{props.minor}",
            "sm_count": props.multi_processor_count,
            "l2_bytes": l2_bytes(),
            "read_ceiling_gb_s": READ_CEILING_GB_S,
            "clocks_locked": False,   # denied on consumer GeForce; hardware.md
            "driver_mode": "WDDM",
        },
        "versions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
        "config": {
            "M": M, "warmup": args.warmup, "iters": args.iters,
            "n_copies_rule": f"max({MIN_COPIES}, ceil({L2_MULTIPLE} * L2 / bytes))",
        },
        "rows": [],
    }

    print(f"{props.name}  L2 {l2_bytes() / 2 ** 20:.0f} MiB  "
          f"torch {torch.__version__}")
    microbench(args, record)
    if args.e2e:
        e2e(args, record)

    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    path = out / f"gemv_bench_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"wrote {path.relative_to(path.parent.parent)}")


if __name__ == "__main__":
    main()
