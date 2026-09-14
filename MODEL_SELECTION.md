# Model selection for the bandwidth-bound decode benchmark

Decision record for which model(s) to use so that "achieved DRAM bandwidth as %
of theoretical peak" (OUTLINE §9) is actually measuring bandwidth, not Python
dispatch. Produced using the verified analytical model in `dequant/byte_model.py`
via `scripts/decode_bytes_report.py` and a throwaway evaluation script against
several candidates' real `config.json` files. No model weights were downloaded;
only config files (a few KB each) were fetched.

---

## 1. The problem, restated quantitatively

Measured on GPT-2 124M (fp16, batch 1, HF eager, RTX 5060 Ti):

| | value |
|---|---|
| Measured decode latency | 9.7 ms/token (p25 9.6, p75 9.9) |
| Analytical bandwidth floor | 0.55 ms/token |
| Ratio (measured / floor) | **17.6x** |
| Achieved bandwidth | ~5.7% of 448 GB/s peak |

So ~94% of the GPT-2 decode step is fixed cost that has nothing to do with
memory traffic: kernel launches, Python module dispatch inside HF's eager
forward, CUDA stream/context overhead. That fixed cost is incurred once per
*layer* (each `GPT2Block` issues a roughly constant number of kernel launches
and Python calls, independent of how wide the layer is), while the useful
signal — bytes read from DRAM — scales with total parameter bytes, which is
`n_layers × bytes_per_layer`. So the metric that actually predicts how close a
model gets to bandwidth-bound is:

```
ratio_to_floor ≈ 1 + (overhead_per_layer × n_layers) / (bytes_per_layer × n_layers / achieved_BW)
              = 1 + overhead_per_layer / (per_layer_floor_time)
```

Concretely: **a model with fewer, wider layers is better than one with more,
narrower layers at the same total parameter count**, because overhead scales
with layer *count* while the floor scales with layer *width* (roughly
`hidden × intermediate`). This is why Qwen2.5-1.5B (28 layers, hidden 1536)
is predicted to be *worse* than Llama-3.2-1B (16 layers, hidden 2048) despite
having more parameters — see the table below.

### Deriving a per-layer overhead estimate from the one data point we have

GPT-2 124M has 12 layers. Attributing all of the measured overhead to the
layer loop (see §4 for why this is a simplification):

```
total_overhead   = 9.7 ms − 0.55 ms = 9.15 ms
overhead_per_layer ≈ 9.15 ms / 12 layers ≈ 0.76 ms/layer
```

This number — **~0.76 ms of launch/dispatch overhead per transformer layer,
on this machine, in HF eager mode** — is the estimate used to predict every
other candidate below: `est_overhead_ms = 0.76 × n_layers`,
`est_measured_ms = floor_ms + est_overhead_ms`. It is a single-point linear
extrapolation, stated as such, and it is the single biggest source of
uncertainty in this document (§4).

---

## 2. Candidate comparison

Floor and byte counts computed with `dequant/byte_model.py` at `seq_len=128`
(short context, matching the regime the 0.55 ms/token figure was measured at;
floor barely moves with context for these model sizes since weights + LM head
dominate over KV traffic — see the full sweep from
`scripts/decode_bytes_report.py` for the longer-context trend). VRAM budget:
RTX 5060 Ti, 16311 MiB ≈ 17.1 GB total. All fp16-weight sizes below were
cross-checked against each model's real `model.safetensors.index.json` /
`Content-Length` on HuggingFace and matched the byte-model prediction to
within rounding, which is a useful independent sanity check on the byte model
itself.

| Model | Params | Layers | Hidden | fp16 weights | Floor ms/tok | Est. measured ms/tok | Est. ratio-to-floor | Fits 16 GB? | Gated? | Download |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-2 124M *(reference, measured)* | 124M | 12 | 768 | 0.25 GB | 0.55 (measured) | **9.7 (measured)** | **17.6x** | yes | no | ~0.5 GB |
| Llama-3.2-1B-Instruct | 1.24B | 16 | 2048 | 2.47 GB | 5.53 | 17.7 | **3.2x** (~31% peak) | yes, ~14.6 GB headroom | **yes** (meta-llama) | ~2.5 GB |
| Qwen2.5-1.5B-Instruct | 1.54B | 28 | 1536 | 3.09 GB | 6.91 | 28.3 | **4.1x** (~25% peak) | yes, ~14.0 GB headroom | no | ~3.1 GB |
| Qwen2.5-3B-Instruct | 3.09B | 36 | 2048 | 6.17 GB | 13.80 | 41.3 | **3.0x** (~33% peak) | yes, ~10.9 GB headroom | no | ~6.2 GB |
| **Llama-3.2-3B-Instruct** (via `unsloth/Llama-3.2-3B-Instruct` mirror) | 3.21B | 28 | 3072 | 6.43 GB | 14.39 | 35.7 | **2.5x** (~40% peak) | yes, ~10.7 GB headroom | underlying repo yes; mirror **no** | ~6.4 GB |
| Mistral-7B-Instruct-v0.3 | 7.25B | 32 | 4096 | 14.50 GB | 31.82 | 56.2 | 1.8x (~57% peak) | **NO** — only ~2.6 GB headroom | no | ~14.5 GB |
| Qwen2.5-7B-Instruct | 7.62B | 28 | 3584 | 15.23 GB | 31.60 | 53.0 | 1.7x (~60% peak) | **NO** — only ~1.9 GB headroom | no | ~15.2 GB |
| (Llama-3.1-8B, for reference) | 8.03B | 32 | 4096 | 16.06 GB | — | — | ~1.7x (extrapolated) | **NO** — ~1.0 GB headroom | yes | ~16.1 GB |

Notes on the table:

- **"Fits 16 GB?" headroom** is fp16 weights subtracted from ~17.1 GB total,
  before KV cache, activations, and CUDA context (typically 0.3–1 GB just for
  the context + allocator, more once KV cache grows at long context). The two
  7B-class models leave under 3 GB for all of that — real OOM risk, especially
  during prefill where HF eager attention materializes O(seq_len²) score
  matrices. They are **rejected** on this constraint, not on ratio.
- The 7B/8B rows *would* give the best ratio-to-floor if they fit — this is
  the direct tension the outline anticipates: bigger is more bandwidth-bound,
  but this card caps how big "bigger" can be.
- **Qwen2.5-1.5B-Instruct is predicted to be the worst of the fitting
  candidates**, worse even than the smaller Llama-3.2-1B, because it is a
  28-layer, narrow-hidden (1536) design. It is direct evidence for the
  bytes-per-layer-vs-overhead-per-layer argument in §1: more layers at similar
  or lower per-layer width loses on this metric even at a higher total
  parameter count.
- `unsloth/Llama-3.2-3B-Instruct` was verified to host full safetensors
  weights (not just a 4-bit/adapter variant): `model.safetensors.index.json`
  reports `total_size = 6,425,499,648` bytes, matching the byte-model
  prediction almost exactly. `config.json` fetches with a plain HTTP GET (200,
  no token) — confirmed ungated, unlike `meta-llama/Llama-3.2-3B-Instruct`
  itself (401 without a token, confirmed by curl). `mistralai/Mistral-7B-Instruct-v0.3`
  is also confirmed ungated (200 via redirect) — it's excluded on VRAM, not gating.

---

## 3. Recommendation

### Primary: Llama-3.2-3B-Instruct, via the `unsloth/Llama-3.2-3B-Instruct` ungated mirror

- Predicted ratio-to-floor **~2.5x (~40% of peak bandwidth)** — a ~7x
  improvement over GPT-2's 17.6x, and meaningfully better than either model
  OUTLINE §2 currently names (3.2x / 4.1x).
- Same architecture family (`llama`, GQA, RMSNorm, SwiGLU MLP) as the model
  already planned as primary in OUTLINE §2 — the packer, kernel, and format
  sweep designed for Llama-3.2-1B need no rework, only new per-layer
  dimensions (hidden 3072, intermediate 8192, kv_heads 8×128 vs 1B's
  8×64 — recompute the per-layer byte budget in §5-style tables before
  Phase 1).
- Fits with ~10.7 GB of real headroom for KV cache, activations, and CUDA
  context at this scale — comfortable, not borderline.
- Gating is sidestepped via the `unsloth` mirror. If HF gating access is later
  granted (HF_TOKEN + license acceptance on meta-llama), switch to
  `meta-llama/Llama-3.2-3B-Instruct` directly — same weights, no other change.
- ~6.4 GB download.

### Fallback: Qwen2.5-3B-Instruct

- Fully ungated, official Qwen namespace — no mirror-trust question at all.
- Predicted ratio-to-floor ~3.0x (~33% of peak) — worse than the primary
  because it is deeper (36 vs 28 layers) at similar total size, but still a
  clear improvement over the current OUTLINE fallback (Qwen2.5-1.5B, 4.1x).
- ~6.2 GB download, ~10.9 GB headroom.

### On the models OUTLINE §2 currently names

**They are not adequate for a convincing bandwidth claim**, though they are
not as broken as GPT-2. Llama-3.2-1B (3.2x, ~31% of peak) and especially
Qwen2.5-1.5B (4.1x, ~25% of peak) still spend the majority of the decode step
on overhead by this estimate. Keep GPT-2 as the fast-iteration/correctness
model (OUTLINE §2's stated role, unchanged) and keep Llama-3.2-1B and
Qwen2.5-1.5B in the format sweep for the model-size-sensitivity axis (OUTLINE
§9's "quantization sensitivity vs. model size" metric genuinely wants small
models) — but do not use either of them as the number that goes in the
headline "% of peak bandwidth" claim. Use the 3B-class model for that.

---

## 4. Limitations — where this estimate could be wrong

1. **Single data point.** The 0.76 ms/layer figure comes from exactly one
   measurement (GPT-2, 12 layers). It cannot distinguish a true per-layer
   cost from a fixed per-decode-step cost (embedding lookup, final norm, LM
   head launch, sampling, `generate()`-loop Python bookkeeping) that doesn't
   scale with layer count at all. If a meaningful chunk of the 9.15 ms is
   fixed rather than per-layer, the true per-layer slope is smaller than
   0.76 ms, and every ratio predicted here for higher-layer-count models
   (Qwen's 28–36 layers) is an **overestimate** — those models would look
   relatively better than this table suggests, and the gap between Llama-3.2
   and Qwen would narrow.
2. **Architecture-dependent launch count.** GPT-2's layer and a Llama/Qwen
   layer do not issue the same number of kernels/Python calls. Llama-family
   attention adds RoPE application and, for GQA models, a `repeat_kv`
   expansion that GPT-2 (plain MHA, no RoPE) doesn't have — pushing per-layer
   overhead up. Conversely, if HF's SDPA backend fuses attention into fewer
   kernel launches than GPT-2's eager path, that pushes it down. Net direction
   is not resolved by this analysis.
3. **Floor itself is idealized.** `decode_bytes_report.py` assumes 100% of
   theoretical peak bandwidth is achievable. OUTLINE §3 notes a competent
   GEMV kernel realistically sits at 80–90% of peak, so even a genuinely
   bandwidth-bound decode step will show ratio ≈ 1.1–1.25x, not exactly 1.0x.
   Don't chase 1.0x as a literal target.
4. **KV cache growth is not exercised at seq_len=128.** All ratios above are
   computed near the start of generation. At long context (32k), KV traffic
   becomes non-trivial for GQA models (see `decode_bytes_report.py`'s
   seq_len sweep) and shifts the floor upward independent of the model choice
   — a secondary, second-order effect on the ratio not captured in this
   table's single-context snapshot.

**What would settle this**: benchmark actual decode latency (same
methodology as the GPT-2 measurement) on at least two more real models at
different layer counts — e.g., Llama-3.2-1B (16 layers) and the recommended
Llama-3.2-3B (28 layers) — and fit `overhead_ms = a + b × n_layers` across
the (now three) data points. That separates the fixed and per-layer terms
directly instead of assuming the fixed term is zero, and it calibrates the
architecture correction in point 2 above using real Llama-family numbers
instead of extrapolating from GPT-2's.

---

## 5. Does fixing overhead change the answer, instead of a bigger model?

**Both — they're not substitutes, and they act at different points in the
plan.**

- **A bigger model is the free win, and it's Phase-0-cheap.** Swapping
  Llama-3.2-1B → Llama-3.2-3B requires no new code, just a config/checkpoint
  change, and predicted ratio-to-floor drops from 3.2x to 2.5x (~31% → ~40%
  of peak) using this estimate. Do this immediately — before any kernel work
  — because Phase 0's "measure launch-overhead fraction of a decode step"
  deliverable is only informative relative to a floor the model is close
  enough to for the comparison to mean something.
- **But even the 3B-class recommendation only gets to ~2.5x, not ~1.1x.**
  Reaching a ratio where "% of peak bandwidth" is a tight, convincing number
  (not "40% of peak, mostly overhead-limited") requires cutting launch count,
  which is exactly what OUTLINE Phase 2 (kernel fusion: gate/up/SwiGLU fused,
  RMSNorm fused into the following projection — cuts kernels per layer) and
  Phase 5 (CUDA Graph capture, closing out the rest of the launch-overhead
  story) already plan to do. Per the bytes-per-layer argument in §1, fusion
  directly attacks the `overhead_per_layer` term the same way a bigger model
  attacks the `bytes_per_layer` term — they're two knobs on the same ratio.
- **Model size alone cannot reach ratio≈1.1x on this card.** The only
  candidates predicted to get there (Mistral-7B, Qwen2.5-7B, ~1.7–1.8x, still
  not quite there) don't fit in 16 GB with real headroom. So "just use a
  bigger model" runs out of headroom before it runs out of need — overhead
  reduction is not optional if the headline metric is meant to be tight.
- **Practical sequencing implication for this project**: adopt the 3B model
  now for a more honest Phase 0 baseline, but also consider pulling forward a
  cheap version of overhead reduction — HuggingFace's
  `cache_implementation="static"` plus `torch.compile`, both usable before any
  custom CUDA graph capture work — as an early, low-effort Phase 0/1 step to
  get a first believable "% of peak" number, ahead of the full CUDA Graph
  capture that OUTLINE correctly defers to Phase 5 (it needs the paged KV
  allocator from Phase 3 first for static addresses).
