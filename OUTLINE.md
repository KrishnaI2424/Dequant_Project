# In-Register Dequantization for Memory-Bound LLM Decode

A CUDA project plan. Target: one semester (~14 weeks), single consumer GPU.

**Centerpiece**: W4A16 weight dequantization, characterized across format space and
fused with surrounding ops.
**Extension**: the same technique applied to the KV cache, where quantization moves
onto the critical path.

---

## 1. The thesis

The unifying technique is **never materializing dequantized values in DRAM**. You load
a compressed representation, expand it in registers, consume it immediately, and throw
it away. The moment you write dequantized values back to memory to hand them to a
library GEMM, you have moved 4× the bytes and lost the entire benefit.

That technique has two targets with very different constraints:

| | Weights | KV cache |
|---|---|---|
| When quantized | Offline, once | Online, every token |
| Runtime quantization cost | Zero | On the critical path |
| Format | Fixed at pack time | Per-token scales known at append; per-channel K scales need a full token group |
| Difficulty | Moderate | Hard |

The weights case is the core of the project. The KV case is the extension that shows
the technique generalizes to a setting where the easy assumptions break.

**What this project claims**: a characterization of the format space (bit width × group
size × symmetry × model size) against achieved bandwidth and accuracy, on consumer
hardware, plus a fusion study of what can be folded into the dequant kernel.

**What it does not claim**: peak throughput. See §3.

---

## 2. Models

### Primary: Llama-3.2-1B-Instruct

| Property | Value |
|---|---|
| Layers | 16 |
| Hidden dim | 2048 |
| Intermediate dim | 8192 |
| Q / KV heads | 32 / 8 (GQA) |
| Head dim | 64 |
| Native context | 128k |
| Weights (fp16) | ~2.5 GB |

Per-layer parameter budget, which drives every bandwidth calculation:

- Attention: `q_proj` 4.2M + `k_proj` 1.05M + `v_proj` 1.05M + `o_proj` 4.2M = **10.5M**
- MLP: `3 × 2048 × 8192` = **50.3M**
- Total per layer: ~60.8M params → ~122 MB fp16, ~30 MB at INT4

The MLP is 83% of each layer's weight traffic. That is where the kernel time is, and
it is why Phase 2 targets the MLP specifically.

Fallback if HuggingFace gating is annoying: **Qwen2.5-1.5B-Instruct**, ungated.

### Secondary: GPT-2 small via nanoGPT (124M)

Two jobs:

1. **Fast iteration sandbox.** You already know the codebase from the RMSNorm
   integration. Debug kernels here where a full forward pass is near-instant.
2. **Small-model contrast.** Quantization damage grows as models shrink — a 124M model
   at INT4 g128 will degrade visibly where a 1B model barely moves. Measuring that
   sensitivity across 124M / 1B / 1.5B is a real finding and costs you almost nothing,
   since the packer and kernel are already parameterized.

Note the weight checkpoints ship in fp32 or bf16, not fp16. You cast down yourself, and
you keep the fp32 originals as quantization-error ground truth.

---

## 3. Positioning against Marlin

This has to be settled before you write the README, not after someone asks.

At batch 1 the decode step is bandwidth-bound: runtime ≈ weight bytes ÷ achieved
bandwidth. Every INT4 kernel moves the same bytes. A competent GEMV already sits at
80–90% of peak, and in recent head-to-head comparisons of W4A16 / W4A8 / W4A4 kernels,
Marlin — with the lightest dequantization path — comes out fastest at small batch. There
is no meaningful performance headroom at batch 1, for you or for anyone.

So the framing is not "a fast W4A16 kernel." It is:

> I implemented W4A16 dequant-GEMV from scratch, made the quantization format a kernel
> parameter, measured the full format space against achieved bandwidth and accuracy
> across three model scales, and studied what can be fused into the dequant kernel.

That claim is true, defensible, and not something Marlin makes. Be precise about what
Marlin does and doesn't cover, because an interviewer will be: the vLLM Marlin family
supports 4-bit, 8-bit, and FP8 weights, symmetric and (via the AWQ variant) asymmetric,
with group sizes restricted to {32, 64, 128, per-channel}, tile-aligned dimensions
(N a multiple of 64, K a multiple of 128), and sm80+ only. It does **not** cover sub-4-bit
widths, other group sizes, arbitrary shapes, or alternative packing layouts — and it is
not built to answer "what does each of those choices cost." That narrowness is what makes
it fast. A parameterized characterization kernel is the opposite artifact and answers a
different question.

The thing being tested in an interview is whether you can explain *why* Marlin overlaps
dequantization with memory access the way it does. Building your own is the only
reliable path to being able to answer that.

---

## 4. Stack

- **GPU**: your dual-GPU desktop, all measurements single-GPU. Record SM count, memory
  clock, and theoretical peak bandwidth — every number is normalized against these.
- **CUDA**: nvcc targeting your cards' actual compute capability. Avoid sm80+-only
  paths unless both cards support them.
- **Host**: PyTorch with `torch.utils.cpp_extension` (same toolchain as the RMSNorm
  work), HuggingFace `transformers` for checkpoints and the fp16 reference.
- **Profiling**: Nsight Compute for `dram__bytes`, sector counts, occupancy; Nsight
  Systems for launch overhead and timeline gaps.
- **Reproducibility**: pin CUDA toolkit, driver, PyTorch, transformers versions in the
  README. Benchmark numbers without version pins are not results.

---

## 5. Weight processing (offline)

### What gets quantized

Quantize: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

Leave alone: RMSNorm scales and biases. Tiny, and quantizing them costs accuracy for
nothing.

### The LM head problem — read this before deciding what to skip

The usual advice is "leave the embedding and LM head in fp16." For a bandwidth study on
small models that advice is a trap, and this plan nearly walked into it.

Llama-3.2-1B ties its embedding to its LM head: one `128256 × 2048` tensor, **525 MB in
fp16**, read in full by the final GEMV on every decode step. After you quantize the 16
transformer layers to INT4 they total roughly **487 MB**. The unquantized LM head is now
the single largest term in the decode step — more than all sixteen layers combined.

GPT-2 small is the same story: ~42 MB of INT4 layers against a ~77 MB fp16 tied head.

This is Amdahl's law applied to quantization, and it is a real finding, not a footnote:
on small models, the honest end-to-end speedup from W4A16 is capped hard by the LM head
until you deal with it. Treat it as its own experiment:

1. Report the LM head's share of decode bytes in Phase 0, before any quantization.
2. Quantize the LM head **separately** — INT8 first, which is typically near-lossless
   for this tensor, then INT4 — and measure its accuracy impact in isolation from the
   layer quantization. Vocabulary logits are sensitive to different failure modes than
   hidden-state projections, so do not fold this into the layer sweep.
3. Because the tensor is tied, quantizing it affects the input embedding lookup too.
   Decide whether to keep an fp16 copy for the lookup (costs 525 MB of memory, no decode
   bandwidth — the lookup reads one row) or share the quantized tensor. Measure both.

Nobody publishes this cleanly for sub-2B models on consumer cards. It is one of the more
distinctive things this project can say.

### Format parameters — all of these are packer inputs, not constants

| Parameter | Range to sweep |
|---|---|
| Bit width | 2, 4, 8 (3-bit as stretch — it doesn't pack cleanly into 32-bit words) |
| Group size | 32, 64, 128, 256, per-channel |
| Symmetry | symmetric, asymmetric |
| Packing layout | sequential, interleaved |

### Symmetric (default for weights)

```
scale = max(|w_group|) / (2^(b-1) - 1)
q     = clamp(round(w / scale), -2^(b-1), 2^(b-1) - 1)
w'    = q * scale
```

No zero-point stored, no subtract in the inner loop. Weights cluster around zero, so
this is the right default.

### Asymmetric

```
scale = (max(w_group) - min(w_group)) / (2^b - 1)
zp    = round(-min(w_group) / scale)
q     = clamp(round(w / scale) + zp, 0, 2^b - 1)
w'    = (q - zp) * scale
```

Costs an extra subtract per element and an extra stored value per group. Include it in
the sweep so you can report what that costs — that measurement is part of the
contribution.

### Packing

Two INT4 per `uint8`, or eight per `uint32` (which loads better — prefer 32-bit or
128-bit loads over byte loads).

**Layout matters and is worth sweeping.** Sequential packing (`q0,q1` in byte 0) means
unpacking element *i* and *i+1* together. Interleaved packing (`q0,q4` in byte 0) can
reduce the shift/mask instruction count when you unpack eight values at once from a
`uint32`. This is a real, measurable difference in the inner loop and nobody documents
it clearly for consumer cards.

### Serialization

Write packed weights + fp16 scales (+ zero-points if asymmetric) into one checkpoint,
with bit width, group size, symmetry, and packing layout recorded in the metadata.
**Layout mismatch between packer and kernel is the number-one source of silent wrong
answers in this project.** Make the kernel assert the metadata matches its compiled
configuration.

---

## 6. Kernel design notes

Not code — the decisions you'll be making.

- **Parallelization**: one warp per output row is the natural starting point. Each lane
  strides across the input dimension, accumulating partial dot products, then a
  warp-shuffle reduction collapses to one value. Consider multiple warps per row for
  large input dims to improve occupancy.
- **Loads**: vectorized. A `uint4` load pulls 128 bits = 32 INT4 weights in one
  instruction. This is the single biggest lever on achieved bandwidth.
- **Unpack**: mask and shift per nibble, then sign-extend. For signed INT4 the common
  trick is XOR with 0x8 then subtract 8, avoiding a branch. Count the instructions —
  at batch 1 you have compute to spare, but not unlimited.
- **Scales and lane mapping** — a real design decision, not a detail. Two layouts:
  (a) lanes stride *within* a group, so one scale is broadcast to the whole warp per
  group boundary, but loads are narrow; (b) each lane owns a contiguous 128-bit chunk
  (32 INT4s), so loads are wide but one warp-wide load spans ~1024 elements = 8 groups
  and every lane needs its own scale. Wider loads win on bandwidth; per-lane scales cost
  registers and a gather. This tradeoff shifts with group size and is part of what the
  sweep should expose. Don't assume; measure both.
- **Accumulate in fp32**, not fp16, then cast at the end. fp16 accumulation error over
  a 2048- or 8192-element dot product is not negligible and will contaminate your
  quantization-error attribution.

**What to measure in Nsight**: `dram__bytes_read` against your analytical byte count
(if they disagree you have a caching or coalescing problem), achieved occupancy, and
warp stall reasons. For a memory-bound kernel, "long scoreboard" stalls dominating is
the expected and healthy signal.

---

## 7. Where each piece sits in inference

### Prefill (one pass over all N prompt tokens)

```
embed lookup
per layer × 16:
    RMSNorm                              ← existing kernel
    Q,K,V projections (GEMM, M=N)        ← dequant weights, Phase 1
    RoPE on Q and K
    causal attention over all N tokens
    write K,V to cache                   ← Phase 4
    O projection (GEMM)                  ← dequant weights
    residual
    RMSNorm                              ← existing kernel
    MLP: gate/up (GEMM), SwiGLU, down    ← Phase 2 fusion target
    residual
final RMSNorm → LM head → sample token 1
```

Compute-bound and GEMM-shaped. Not the optimization target, but your dequant path must
be correct here too.

### Decode (per generated token)

```
embed lookup (1 token)
per layer × 16:
    RMSNorm                              ← existing kernel
    Q,K,V projections (GEMV)             ← Phase 1 — 10.5M params
    RoPE on Q and new K
    append K,V to cache                  ← Phase 4 (quantize on write)
    attention: read cache, dot, softmax  ← Phase 4 (dequant in registers)
    O projection (GEMV)                  ← Phase 1
    residual
    RMSNorm                              ← existing kernel
    MLP: gate/up GEMV, SwiGLU, down GEMV ← Phase 1 + Phase 2 — 50.3M params
    residual
final RMSNorm → LM head (GEMV) → sample   ← 525 MB fp16; see §5 before ignoring it
```

By Phase 4 you own every memory-bound kernel in this loop.

---

## 8. Phases

### Phase 0 — Instrumentation and roofline (2 weeks)

Before any kernel.

- Llama-3.2-1B running in fp16 through HuggingFace. Correctness baseline established.
- Analytical model: bytes read per decode step, split into transformer-layer weights
  (fixed), LM head (fixed, and large — see §5), and KV (grows with sequence length).
- Measure decode latency vs. sequence length. Plot predicted vs. measured.
- Measure achieved bandwidth for the fp16 baseline — this is the number every later
  result is compared against.
- Measure the launch-overhead fraction of a decode step (Nsight Systems, kernel time
  vs. wall time). This sizes the Phase 2 opportunity.

**Deliverable**: the roofline plot plus a launch-overhead number. If measured and
predicted diverge, stop and find out why — everything downstream assumes this model is
trustworthy.

### Phase 1 — W4A16 dequant-GEMV and the format sweep (4 weeks) — the core

- Offline packer per §5, fully parameterized.
- Kernel per §6.
- Validate against fp16 reference: max absolute error, relative error distribution,
  per-layer. Differential-test at every shape the model actually uses.
- Integrate as a PyTorch custom op, swap into the model, verify generation quality.
- **The sweep**: bit width × group size × symmetry × packing layout → achieved
  bandwidth utilization and perplexity delta. Repeat across GPT-2 124M, Llama-1B, and
  Qwen-1.5B for the model-size sensitivity axis.

**Deliverable**: the format-space characterization. This is the project's headline
result — a map of what each format choice costs in bandwidth and in accuracy, measured
rather than asserted.

### Phase 2 — Fusion (2 weeks)

An honest framing point first: at batch 1, activation traffic is tiny next to weight
traffic. For Llama-1B the gate and up outputs are 16 KB each; fusing gate/up/SwiGLU
avoids writing both, reading both back, and round-tripping the SwiGLU output — about
**64 KB saved per layer against ~30 MB of INT4 weights, roughly 0.2%**. Fusion's payoff
here is not bandwidth. It is:

1. **Launch count.** 16 layers × ~8 kernels × hundreds of tokens. Phase 0 measured how
   much of the decode step is launch overhead; this is the first of two attacks on it
   (the second, CUDA Graphs, is in Phase 5 — see there for why it waits).
2. **Scaling headroom.** Activation traffic grows with batch size; weight traffic does
   not. The fusion that looks pointless at batch 1 matters at batch 8+. Measure across
   batch sizes and show the trend — that curve is the deliverable.

Work:

- Fuse `gate_proj` + `up_proj` + SwiGLU into one dequant kernel. Both projections read
  the same input vector, so you load `x` once. Leave `down_proj` separate — it needs the
  complete 8192-wide intermediate, so fusing it requires grid-wide sync.
- Fuse RMSNorm into the following projection. The RMS reduction over 2048 elements is
  cheap enough to recompute per block rather than broadcast.
- Optional stretch: persistent megakernel for one full layer using cooperative groups.
  Expect it to lose on occupancy; report that if so.

**Deliverable**: kernel count before/after, and the fusion benefit curve as a function
of batch size.

### Phase 3 — KV cache extension: paged allocator (2 weeks)

- Block allocator over a preallocated device pool. Fixed-size blocks (16/32/64/128
  tokens), free list, allocate and release.
- Block table per sequence: logical position → physical block + offset.
- Decode attention kernel gathering K,V through the block table. Single query token, so
  this is dot products + softmax + weighted sum — no tiled flash attention needed.
- Correctness against the contiguous-cache reference.
- Measure fragmentation vs. contiguous, and gather overhead vs. block size.

**Deliverable**: block-size sweep showing the fragmentation/overhead tradeoff.

### Phase 4 — KV cache extension: online quantization (2 weeks)

The point of this phase is demonstrating that the Phase 1 technique survives when
quantization moves onto the critical path.

- **RoPE ordering is an experimental axis, not a decision.** The earlier draft of this
  plan said "quantize K after RoPE, never before." That was wrong. RoPE rotates channel
  pairs by position-dependent angles, which smears K's outlier channels into their
  neighbours and partly dissolves the fixed-channel structure per-channel quantization
  relies on. KVQuant reports pre-RoPE per-channel key quantization beating post-RoPE by
  ~0.8 perplexity at 3-bit on LLaMA-7B, and builds a fused kernel that dequantizes the
  stored pre-RoPE key and applies the rotation on the fly during attention. Post-RoPE is
  the better choice only when quantizing keys per-token, which this plan doesn't do.
  There is no "requantize" step either way — the cost of pre-RoPE storage is one
  rotation per cached key per read, which is compute you have to spare at batch 1 and
  may not have at higher batch. So: implement both. Measure accuracy and per-read
  compute for each on your hardware. The crossover between them is a result nobody has
  published for this model class.
- Llama-3.2 uses the `llama3` RoPE scaling variant (factor 32, frequency-dependent
  adjustments). Whichever ordering you use, the fused kernel must reproduce that exact
  frequency schedule or the model silently degrades.
- **Measure the K/V asymmetry first.** Plot per-channel magnitude distributions for K
  and V at several layers. K typically has persistent outlier channels; V is better
  behaved. Design around what you measure, not what the literature says.
- Per-token scaling for V. For K, per-channel scaling over a window of recent tokens,
  with the most recent tokens held unquantized in a small fp16 buffer.
- **Measure the online quantization cost in isolation before building the fused
  kernel.** If quantize-on-append costs more than the bandwidth it saves, you need to
  know in week 1 of this phase, not week 2.
- Fused dequant-attention: gather packed KV through the block table, expand in
  registers.
- **INT8 first.** It is often near-lossless and may be the honest recommendation. INT4
  KV is stretch.
- Accuracy: perplexity plus a long-context retrieval task. Perplexity alone hides
  long-range degradation, which is precisely what KV quantization damages.

**Deliverable**: decode latency vs. sequence length for fp16 / INT8 / INT4 KV, with the
accuracy cost stated alongside.

### Phase 5 — CUDA Graphs, integration, and writeup (2 weeks)

CUDA Graph capture lives here rather than in Phase 2 for a specific reason: graphs
require **static memory addresses and static shapes**, and the decode step's attention
kernel has a loop bound that grows by one every token. Capturing before the paged cache
exists means re-doing it after. Capturing once the allocator is in place means the
preallocated pool and device-resident block table are already graph-safe, and the only
remaining issue is sequence length. Handle that one of three ways — capture one graph
per length bucket and replay the matching one, capture at max length with masking, or
have the kernel read the current length from device memory — and report the cost of
whichever you pick. HuggingFace's own `cache_implementation="static"` exists for exactly
this reason.

- CUDA Graph capture of the decode step; measure against the ungraphed baseline. Combined
  with Phase 2's fusion, this closes out the launch-overhead story.
- End-to-end tokens/sec with all kernels active.
- Nsight profiles for the dequant-GEMV: achieved bandwidth, occupancy, where the
  remaining gap to peak lives.
- README with methodology, hardware spec, version pins, every plot.
- One command regenerates each figure.

---

## 9. Metrics

| Metric | Why |
|---|---|
| Achieved DRAM bandwidth as % of theoretical peak | The only meaningful efficiency number for a memory-bound kernel |
| LM head share of decode bytes, before and after layer quantization | The Amdahl bound on end-to-end speedup for small models |
| Perplexity delta vs. fp16, per format | Core of the characterization |
| Format sweep grid (bandwidth × accuracy) | The headline result |
| Quantization sensitivity vs. model size | Shows the finding generalizes (or doesn't) |
| Launch overhead as % of decode step | Phase 2 motivation and result |
| Fusion benefit vs. batch size | Honest accounting of where fusion pays |
| Decode latency vs. sequence length | Phase 3/4 result |
| Quantize-on-append cost as % of decode step | The online-quantization overhead |

Report medians and spread over repeated runs. Lock clocks, or state that you didn't.

---

## 10. Limitations to state in the writeup

Stating these yourself is worth more than having them found.

- **No performance headroom at batch 1.** Marlin and similar kernels already sit near
  the bandwidth roof. This project does not beat them and does not claim to.
- **Batch 1 focus.** Phase 2's batch sweep is the only part that speaks to batched
  serving. No continuous batching, no preemption, no prefix sharing.
- **Single GPU, consumer hardware.** No NVLink. Results do not extrapolate to
  datacenter topology.
- **Small models.** Findings at 124M–1.5B may not hold at 70B, where layer counts, head
  configurations, and outlier behavior all differ. The model-size axis partially
  addresses this but cannot fully.
- **Weight quantization is a saturated area.** GPTQ, AWQ, Marlin, QQQ, Machete, and
  others exist. The contribution is characterization and fusion, not novelty.
- **KV quantization is also not novel** (KIVI, KVQuant, fp8 KV in vLLM). Less saturated,
  but not empty.
- **fp16 accumulation error** in attention is separate from quantization error. Separate
  the two when attributing accuracy loss or you will blame the wrong mechanism.
- **INT4 KV may not be viable** at acceptable accuracy. Plan for INT8 being the answer.
- **Fusion savings at batch 1 are ~0.2% of traffic.** The honest justification is launch
  count and batch scaling, not bandwidth.
- **The LM head bounds end-to-end speedup on small models.** Layer-only quantization
  numbers overstate the real gain; report end-to-end with the LM head treated explicitly.
- **RoPE ordering for K is unresolved in the literature for this model class.** Both
  orderings are implemented and measured; neither is assumed correct.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Packing-layout mismatch causes silent wrong answers | Metadata assertions in the kernel; differential test at every phase boundary |
| Format sweep balloons into a combinatorial explosion | Sweep one axis at a time from a fixed baseline; full grid only where axes interact |
| Phase 1 eats the semester | It's the core, so some overrun is acceptable — but hard-stop at week 7 and move to Phase 2 with whatever you have |
| Fused MLP shows no measurable benefit | Expected at batch 1. The batch sweep is the result; report it as such |
| Quantize-on-append costs more than it saves | Measure in isolation in Phase 4 week 1, before building the fused kernel |
| INT4 KV destroys long-context accuracy | INT8 first; INT4 is stretch |
| Can't fit long contexts on the card | Drop to 32k and note the constraint; GPT-2's earlier crossover still supplies the contrast |
| Kernel races found late | `compute-sanitizer` in CI from Phase 1 onward |
| CUDA Graph capture fights the growing sequence length | Deferred to Phase 5; pick a length-bucketing strategy up front and cost it |
| End-to-end speedup looks disappointing after Phase 1 | Expected — the fp16 LM head dominates. Quantize it separately (§5) and re-measure |
| Pre-RoPE fused attention kernel gets the frequency schedule wrong | Differential-test against HuggingFace's RoPE at several positions before quantizing anything |