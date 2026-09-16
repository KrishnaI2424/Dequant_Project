"""Does storing weights in a smaller dtype actually make reading them faster?

The fake-quantization runs could not answer this: they dequantized back to
fp16 before the forward pass, so every format moved identical bytes and
decode latency was flat at 19.05 ms. This measures the underlying claim
directly -- allocate tensors with Llama-3.2-3B's REAL weight shapes at each
storage width, read every byte, and time it.

No model, no matmul, no dequantization. Just: how long does it take the GPU
to pull this many bytes out of DRAM?

To keep the comparison honest, every case is read with the SAME operation:
the raw byte buffer is viewed as fp16 and summed into fp32. Identical kernel
every time, so the only variable is how many bytes exist.

Picking that op mattered. The first version of this script summed uint8 into
int64, which runs at 22.7 GB/s -- 5.9% of peak, i.e. compute-bound on the
accumulator, measuring "time to process N elements" rather than "time to
read N bytes". It produced perfectly linear scaling that looked like a
result but proved nothing about memory. Measured alternatives on this card,
1 GiB buffer:

    uint8 .sum(int64)        22.7 GB/s    <- compute-bound, useless here
    fp16  .sum(float32)     421.8 GB/s    <- saturates read bandwidth
    fp16  .max()            420.6 GB/s
    uint8 copy_ (r+w)       385.4 GB/s

Note the read ceiling (~421 GB/s, 94% of the 448 GB/s theoretical) is HIGHER
than the 384 GB/s copy figure recorded elsewhere in this project as
"practical peak" -- copy pays for reads and writes. Decode is overwhelmingly
weight reads, so ~421 GB/s is the honest ceiling to judge it against.

    python scripts/bandwidth_by_dtype.py
"""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Measured pure-read ceiling on this card (fp16 sum over a 1 GiB buffer).
# Distinct from the 384 GB/s copy figure, which pays for writes too.
READ_PEAK_GB_S = 421.8

# Llama-3.2-3B: 28 layers, hidden 3072, intermediate 8192, kv_dim 1024.
# The seven quantizable projections per layer, as [out, in].
LAYER_SHAPES = [
    (3072, 3072),   # q_proj
    (1024, 3072),   # k_proj
    (1024, 3072),   # v_proj
    (3072, 3072),   # o_proj
    (8192, 3072),   # gate_proj
    (8192, 3072),   # up_proj
    (3072, 8192),   # down_proj
]
N_LAYERS = 28
LM_HEAD = (128256, 3072)

# (label, bytes per logical weight). int4 is stored two-per-byte, which is
# what the packer already produces.
WIDTHS = [
    ("fp16 (today)", 2.0),
    ("int8", 1.0),
    ("int4 packed", 0.5),
]


def build(total_elems, bytes_per_elem):
    """Allocate buffers holding total_elems weights at the given width."""
    n_bytes = int(total_elems * bytes_per_elem)
    # Chunked so no single allocation is absurd; total is what matters.
    chunk = 256 * 1024 * 1024  # even, so the fp16 view is always valid
    bufs, remaining = [], n_bytes
    while remaining > 0:
        take = min(chunk, remaining)
        bufs.append(torch.empty(take, dtype=torch.uint8, device="cuda"))
        remaining -= take
    return bufs, n_bytes


def time_read(bufs, iters=20):
    """Median wall time to read every byte in bufs."""
    def once():
        total = 0
        for b in bufs:
            # View the raw bytes as fp16 and reduce: same kernel for every
            # case, saturates read bandwidth, so only byte count varies.
            total += b.view(torch.float16).sum(dtype=torch.float32)
        return total

    for _ in range(5):
        once()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        once()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def main():
    props = torch.cuda.get_device_properties(0)
    layer_elems = sum(o * i for o, i in LAYER_SHAPES) * N_LAYERS
    head_elems = LM_HEAD[0] * LM_HEAD[1]
    print(f"{props.name}  practical peak {READ_PEAK_GB_S:.0f} GB/s")
    print(f"Llama-3.2-3B weights: {layer_elems:,} layer + {head_elems:,} LM head "
          f"= {layer_elems + head_elems:,} values\n")

    results = []
    print(f"  {'storage':<16}{'MB read':>10}{'ms':>9}{'GB/s':>9}{'%peak':>8}"
          f"{'vs fp16':>9}")

    # Scenario A: quantize the layers, leave the LM head fp16 (OUTLINE s5
    # default). Scenario B: quantize everything including the head.
    for scenario, head_bytes in (("layers quantized, fp16 head", 2.0),
                                 ("layers + head quantized", None)):
        print(f"\n  {scenario}")
        base_ms = None
        for label, w in WIDTHS:
            hb = w if head_bytes is None else head_bytes
            total_bytes = int(layer_elems * w + head_elems * hb)
            bufs, n = build(layer_elems, w)
            hbufs, hn = build(head_elems, hb)
            ms = time_read(bufs + hbufs)
            del bufs, hbufs
            torch.cuda.empty_cache()

            gb_s = total_bytes / (ms * 1e-3) / 1e9
            if base_ms is None:
                base_ms = ms
            results.append({"scenario": scenario, "storage": label,
                            "bytes": total_bytes, "ms": ms, "gb_s": gb_s,
                            "speedup_vs_fp16": base_ms / ms})
            print(f"  {label:<16}{total_bytes / 1e6:>10.0f}{ms:>9.3f}{gb_s:>9.1f}"
                  f"{gb_s / READ_PEAK_GB_S * 100:>7.1f}%{base_ms / ms:>8.2f}x")

    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    path = out / f"bandwidth_by_dtype_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"gpu": props.name,
                                "practical_peak_gb_s": READ_PEAK_GB_S,
                                "results": results}, indent=2))
    print(f"\nwrote {path.relative_to(path.parent.parent)}")


if __name__ == "__main__":
    main()
