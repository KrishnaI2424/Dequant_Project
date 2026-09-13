"""Decode-latency measurement harness.

Phase 0, step 4. This is the apparatus every later result is produced with,
so the measurement discipline lives here rather than in ad-hoc scripts.

Three numbers are recorded per decode step, not one:

    host_ms   wall time for the Python call to RETURN, without syncing.
              This is HuggingFace's dispatch cost -- pure CPU work.
    gpu_ms    CUDA-event elapsed across the step: first kernel start to last
              kernel end, INCLUDING the gaps between kernels.
    total_ms  wall time including the sync. The honest tokens/sec number.

Keeping them apart is the point. On this machine GPT-2 decode is ~17x off its
bandwidth floor, and a single blended number cannot tell you whether that is
memory traffic, kernel launch gaps, or Python. Nsight (step 6) splits gpu_ms
further into kernel time vs gap time; this harness is what tells you whether
that split is even worth chasing.

Holding seq_len fixed
---------------------
A decode step's cost depends on context length, so the sweep is meaningless
unless each timed step runs at exactly one seq_len. A naive loop that keeps
calling the model grows the KV cache by one token per iteration and silently
measures a moving target. Every backend here pins it:

    dynamic  let it append, then crop back OUTSIDE the timed region.
    static   preallocate to exactly seq_len -- not larger, since a static
             cache attends over its whole buffer -- and rewind the write
             counter between iterations so the same slot is rewritten.
    graph    the static path, captured into a CUDA graph. Replay costs almost
             no CPU, so this is the only arm that measures what the GPU can
             actually do. Without it "the GPU is slow" and "Python cannot feed
             the GPU" are indistinguishable: in eager mode the call has not
             even returned before the GPU has gone idle.

Usage:
    python -m dequant.bench --model gpt2
    python -m dequant.bench --model gpt2 --seq-lens 128,512,1024 --iters 100
"""

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache

from dequant.byte_model import decode_bytes, predicted_latency_s, spec_from_hf_config

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Theoretical peak for the RTX 5060 Ti: 14001 MHz x 2 (DDR) x 128-bit / 8.
# See hardware.md. Decimal GB, matching the byte model's decimal MB.
PEAK_BW_GB_S = 448.0


def summarize(samples):
    """Median and spread. Never a mean -- OUTLINE section 9."""
    s = sorted(samples)
    n = len(s)
    return {
        "median": statistics.median(s),
        "p25": s[n // 4],
        "p75": s[(3 * n) // 4],
        "min": s[0],
        "max": s[-1],
        "n": n,
    }


def _prefill(model, cache, n_tokens, device):
    """Fill the cache with n_tokens so the next call is a real decode step."""
    ids = torch.randint(0, model.config.vocab_size, (1, n_tokens), device=device)
    with torch.no_grad():
        model(
            input_ids=ids,
            past_key_values=cache,
            cache_position=torch.arange(n_tokens, device=device),
            use_cache=True,
        )


def time_decode_step(model, seq_len, cache_impl="static", warmup=20, iters=50):
    """Time one decode step at a FIXED context length of seq_len tokens."""
    device = model.device
    token = torch.randint(0, model.config.vocab_size, (1, 1), device=device)
    # The step under test appends token number seq_len, so the cache holds
    # seq_len - 1 before it runs.
    pos = seq_len - 1
    cache_position = torch.tensor([pos], device=device)

    if cache_impl in ("static", "graph"):
        # max_cache_len must equal seq_len exactly: a static cache attends over
        # its whole buffer, so an oversized one would inflate KV traffic.
        cache = StaticCache(config=model.config, max_cache_len=seq_len)
        cache = cache.to(device) if hasattr(cache, "to") else cache
    elif cache_impl == "dynamic":
        cache = DynamicCache()
    else:
        raise ValueError(f"unknown cache_impl: {cache_impl}")

    _prefill(model, cache, pos, device)

    def one_step():
        with torch.no_grad():
            model(
                input_ids=token,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
            )

    def restore():
        # Put the cache back to seq_len - 1 so the next iteration runs at the
        # same context length.
        if cache_impl == "dynamic":
            cache.crop(pos)
        else:
            # StaticLayer.update ignores the cache_position argument and
            # derives its write index from an internal counter it bumps every
            # call. Left alone it walks off the end of the buffer and trips a
            # device-side assert on the second step, so rewind it by hand.
            for layer in cache.layers:
                layer.cumulative_length.fill_(pos)

    if cache_impl == "graph":
        # Capture the step into a CUDA graph, so replay costs almost no CPU.
        # This is what separates "the GPU is slow" from "Python cannot feed
        # the GPU fast enough" -- in eager mode the two are indistinguishable
        # because the call has not even returned before the GPU goes idle.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(5):       # required warmup before capture
                one_step()
                restore()
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        restore()
        with torch.cuda.graph(graph):
            one_step()
        one_step = graph.replay

    for _ in range(warmup):
        one_step()
        restore()
    torch.cuda.synchronize()

    host, gpu, total = [], [], []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for _ in range(iters):
        t0 = time.perf_counter()
        start_ev.record()
        one_step()
        end_ev.record()
        t_ret = time.perf_counter()          # Python returned; GPU may still be busy.
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        host.append((t_ret - t0) * 1e3)
        gpu.append(start_ev.elapsed_time(end_ev))
        total.append((t1 - t0) * 1e3)
        restore()

    return {"host_ms": summarize(host), "gpu_ms": summarize(gpu),
            "total_ms": summarize(total)}


def run(model_id, seq_lens, cache_impls, warmup, iters, dtype):
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                   "fp32": torch.float32}[dtype]

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    model = model.to("cuda").eval()

    spec = spec_from_hf_config(model.config.to_dict(), name=model_id)
    # A model cannot be benchmarked past the context length it can represent.
    seq_lens = [n for n in seq_lens if n <= spec.max_context]

    props = torch.cuda.get_device_properties(0)
    record = {
        "model": model_id,
        "dtype": dtype,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "gpu": props.name,
            "sm": f"{props.major}{props.minor}",
            "sm_count": props.multi_processor_count,
            "peak_bw_gb_s": PEAK_BW_GB_S,
            "clocks_locked": False,   # denied on consumer GeForce; see hardware.md
            "driver_mode": "WDDM",
        },
        "versions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
        "config": {"warmup": warmup, "iters": iters},
        "runs": [],
    }

    print(f"{model_id}  {dtype}  {props.name}  "
          f"(peak {PEAK_BW_GB_S:.0f} GB/s, clocks unlocked)")
    print(f"{'cache':>8}{'seq':>7}{'total ms':>10}{'gpu ms':>9}{'host ms':>9}"
          f"{'pred ms':>9}{'ratio':>7}{'GB/s':>8}{'%peak':>7}")

    for impl in cache_impls:
        for n in seq_lens:
            t = time_decode_step(model, n, impl, warmup, iters)
            predicted_bytes = decode_bytes(spec, n)["total"]
            pred_ms = predicted_latency_s(predicted_bytes, PEAK_BW_GB_S) * 1e3
            measured_ms = t["total_ms"]["median"]
            achieved = predicted_bytes / (measured_ms * 1e-3) / 1e9

            record["runs"].append({
                "cache_impl": impl, "seq_len": n,
                "predicted_bytes": predicted_bytes, "predicted_ms": pred_ms,
                "achieved_gb_s": achieved,
                "pct_peak": achieved / PEAK_BW_GB_S * 100,
                "ratio_to_floor": measured_ms / pred_ms,
                **t,
            })
            print(f"{impl:>8}{n:>7}{measured_ms:>10.3f}{t['gpu_ms']['median']:>9.3f}"
                  f"{t['host_ms']['median']:>9.3f}{pred_ms:>9.3f}"
                  f"{measured_ms / pred_ms:>6.1f}x{achieved:>8.1f}"
                  f"{achieved / PEAK_BW_GB_S * 100:>6.1f}%")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"decode_{model_id.replace('/', '_')}_{dtype}_{stamp}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out.relative_to(RESULTS_DIR.parent)}")
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--seq-lens", default="128,512,1024",
                    help="comma-separated; clamped to the model's max context")
    ap.add_argument("--cache", default="dynamic,static,graph",
                    help="comma-separated: dynamic, static, graph")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    run(
        args.model,
        [int(x) for x in args.seq_lens.split(",")],
        [c.strip() for c in args.cache.split(",")],
        args.warmup,
        args.iters,
        args.dtype,
    )


if __name__ == "__main__":
    main()
