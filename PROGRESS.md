# Progress Log

Running log of what's actually been done, in case a session gets cut short.
Newest entries at the top. See `OUTLINE.md` for the full plan.

---

## Phase 1 — W4A16 dequant-GEMV and the format sweep

### Direct proof that smaller storage widths read faster (`scripts/bandwidth_by_dtype.py`)

The fake-quantization runs (below) could not show a speedup by construction
-- they dequantize back to fp16 before the forward pass, so every format
moves identical bytes. This is a different, narrower experiment that isolates
the one physical claim the whole project rests on: does storing the same
logical weights in a smaller dtype actually reduce DRAM read time? No model,
no matmul, no dequantization -- just allocate Llama-3.2-3B's real weight
volume at each storage width and read every byte.

**First attempt was wrong, and worth recording why.** The initial version
summed the raw bytes as `uint8 -> int64`. It ran at 22.7 GB/s -- 5.9% of
theoretical peak -- meaning it was compute-bound on the int64 accumulator,
not memory-bound at all. It still produced clean 2.00x/4.00x scaling with
byte count, which *looked* like a result. That is precisely the trap: a
compute-bound op scales with element count too, so the wrong measurement can
still look like the right one. Switched to viewing the same raw bytes as
fp16 and reducing to fp32, verified to hit 94.2% of theoretical peak, before
trusting any number from it.

**That also surfaced a real gap in the project's own reference numbers**:
`hardware.md`'s "384 GB/s practical peak" turns out to be a **copy**
benchmark (read + write). Decode is almost entirely weight **reads**. A
pure-read op reaches **421.8 GB/s (94.2% of theoretical)**, ~9% higher than
the copy figure -- large enough to change conclusions, and it already had:
see the correction in the accuracy-reference entry below, where the fp16
decode step's efficiency was 88.7% against the wrong denominator and is
80.8% against the right one. `hardware.md` section 2b now documents both
ceilings and which applies when.

**Result** -- reading Llama-3.2-3B's real weight volume (2,818,572,288 layer
+ 394,002,432 LM head values) at each width, same op, only byte count varies:

| Storage | MB read | ms | GB/s | % read peak | vs fp16 |
|---|---|---|---|---|---|
| fp16 (today) | 6425 | 15.357 | 418.4 | 99.2% | 1.00x |
| int8 | 3607 | 8.647 | 417.1 | 98.9% | **1.78x** |
| int4 packed | 2197 | 5.282 | 416.0 | 98.6% | **2.91x** |

...and with the LM head also quantized (not the section 5 default, but shown
for completeness): int8 **1.99x**, int4 **3.96x** -- against a theoretical
2.00x/4.00x, about as clean as a measurement gets. All three cases sit at
98-99% of read peak, i.e. genuinely DRAM-limited, so the runtime differences
are real and are caused by nothing but byte count.

This is the direct, load-bearing evidence that the whole project's central
claim -- fewer bytes in DRAM means faster decode -- holds physically on this
card, independent of whatever the eventual kernel's overhead turns out to be.
It is also why the fp16-head rows cap at 2.91x rather than 4x: the 788 MB
unquantized LM head becomes 36% of all traffic once the layers shrink to
int4, `OUTLINE.md` section 5's Amdahl argument as a measured number instead
of a prediction.

Results: `results/bandwidth_by_dtype_20260915-183609.json`.

### Accuracy reference on real Llama-3.2-3B weights (`scripts/eval_quantized.py`)

Simulated quantization only -- no dequant-GEMV kernel exists yet (that's still
ahead). Each real `nn.Linear.weight` in the 7 quantizable projection types
(q/k/v/o/gate/up/down; norms and the tied LM head left fp16) is round-tripped
through `dequant/packer.py`'s `quantize_tensor()` -> `dequantize()`, both
plain PyTorch ops, and written back into the model's live GPU weight tensor.
The forward pass afterward is ordinary fp16 matmul -- nothing about the
compute path changes, only the numeric values do. This isolates the
information loss from rounding; it says nothing about real kernel speed,
since the model never leaves DRAM-materialized fp16 the whole way through
(the entire point of the project is a kernel that avoids that -- see
`OUTLINE.md` section 1).

**Methodology bug caught and fixed first**: the first run's perplexity
dataset (wikitext-2) failed to load (`HfUriError` -- the dataset moved to the
`Salesforce/` org; bare `wikitext` fails on `datasets` 5.x) and silently fell
back to repetitive synthetic filler text. Baseline perplexity came out at
1.021 -- the model had no uncertainty left to lose, so quantization damage
couldn't register in the number at all, even though the run "succeeded" with
no error. Removed the fallback entirely; the script now raises rather than
report a number computed on invalid data. Re-run on real wikitext-2 test
(`Salesforce/wikitext`) gives a sane fp16 baseline perplexity of 10.586.

**Weight error validates the packer's own gate.** 13.02% relative error for
INT4 g128 symmetric on real Llama weights vs. the 12.6% the packer's unit
test predicted for synthetic Gaussian data (`scripts/test_packer.py`) --
close enough that Llama's per-group weight distribution is apparently
near-Gaussian.

Results, at seq_len=512 for the byte-model projection (LM head + norms fp16
throughout):

| Format | bits/wt | decode MB | byte speedup | rel err | ΔPPL |
|---|---|---|---|---|---|
| fp16 | 16.125 | 6490 | 1.00x | 0.00% | -- (10.586) |
| INT8 g128 sym | 8.125 | 3716 | 1.75x | 0.72% | +0.013 |
| INT4 g32 sym | 4.500 | 2438 | 2.66x | 10.41% | +0.696 |
| **INT4 g128 asym** | 4.250 | 2350 | **2.76x** | 11.15% | **+0.654** |
| INT4 g128 sym | 4.125 | 2306 | 2.81x | 13.02% | +1.366 |
| INT4 per-channel | 4.000 | 2262 | 2.87x | 18.60% | +7.115 |
| INT2 g128 sym | 2.125 | 1602 | 4.05x | 81.72% | +493510 (broken) |

**Headline finding**: INT4 g128 asymmetric strictly dominates INT4 g32
symmetric -- better perplexity (+0.654 vs +0.696) *and* fewer bytes (4.25 vs
4.50 bits/weight). `OUTLINE.md` section 5 asked for asymmetric's cost to be
measured; measured here, it isn't a cost, it's a net win. INT8 is
effectively free (+0.013 ppl, byte-identical greedy generation) -- supports
quantizing the LM head to INT8 per section 5. Per-channel INT4 is a bad
trade (+7.115 ppl for 0.125 bits/weight saved vs. g128) -- group size is by
far the most sensitive axis. INT2 is not viable as plain round-to-nearest
(ppl ~493k, pure gibberish); section 8's "stretch goal" framing is correct.
Error-to-damage is sharply nonlinear: a 43% increase in weight error
(13.02% -> 18.60%) produced a 5.2x increase in perplexity damage, so
element-wise weight error alone is a poor proxy and the sweep needs
perplexity, not just error norms.

**Decode throughput, measured in the right regime.** `eval_quantized.py`
now calls `dequant.bench.time_decode_step()` per format (batch-1,
seq_len=512, CUDA-graph capture -- the near-roofline regime; eager is
dispatch-bound per the Phase 0 entry below). `%peak` is against the
**measured practical peak of 384 GB/s**, not the 448 GB/s theoretical, and
the byte count is the fp16 one because fake quantization means every format
really does move fp16 bytes:

| Format | decode ms | tok/s | % practical peak |
|---|---|---|---|
| fp16 | 19.055 | 52 | 88.7% |
| INT8 g128 sym | 19.053 | 52 | 88.7% |
| INT4 g32 sym | 19.057 | 52 | 88.7% |
| INT4 g128 asym | 19.061 | 52 | 88.7% |
| INT4 g128 sym | 19.066 | 52 | 88.6% |
| INT4 per-channel | 19.069 | 52 | 88.6% |
| INT2 g128 sym | 19.079 | 52 | 88.6% |

Flat, as it must be -- there is no mechanism by which a format *label* can
change latency when every format is ordinary fp16 by the time the forward
pass runs. Logged because it is now measured rather than assumed, and
because two things fall out of it:

1. **Harness reproducibility is excellent.** 19.055 ms here vs. 19.076 ms
   measured in the Phase 0 decode run days earlier, same seq_len and cache
   arm -- 0.1% apart despite clock locking being unavailable on this card.
   Differences above ~0.5% in future kernel work are therefore real signal,
   not thermal drift. Spread across all seven formats here is 0.14%.
2. **The number the real kernel has to beat: 52 tok/s at 88.7% of practical
   peak.** The fp16 decode path is already near-roofline, so an INT4 kernel
   cannot win on efficiency -- only on moving fewer bytes. Holding the same
   ~340 GB/s achieved against INT4 g128's predicted 2306 MB/step gives
   ~6.8 ms/token, i.e. **~148 tok/s** as the Phase 1 target.

(An earlier version of this entry logged *prefill* throughput from the
perplexity pass instead -- 2048-token windows in single forward calls,
compute-bound GEMM. Wrong regime for this project; replaced by the decode
table above.)

(Correction, see the bandwidth-by-dtype entry below: "384 GB/s practical
peak" above is the wrong denominator for a read-dominated decode step -- it
is a copy figure, which pays for a write as well as a read. The correct
read-only ceiling is 421.8 GB/s, so the fp16 decode step's true efficiency is
**80.8%**, not 88.7%. The ~148 tok/s Phase 1 target is unaffected: it was
derived from decode_bytes_report.py's byte predictions, not from this %peak
figure.)

Results: `results/quant_accuracy_unsloth_Llama-3.2-3B-Instruct_20260915-082450.json`.

**Status**: toolchain, packer (`dequant/packer.py`), and this accuracy
reference are done. Not yet done: the dequant-GEMV kernel itself and the
resulting real bandwidth/latency sweep.

---

## Phase 0 — Instrumentation and roofline

### Decision — Llama-3.2-3B-Instruct is now the primary bandwidth-benchmark model

GPT-2 measured at 5.7% of peak bandwidth eager / 48.5% under CUDA-graph capture --
too small, decode is dispatch-bound not memory-bound. Ran the same harness on
`unsloth/Llama-3.2-3B-Instruct` (ungated mirror; 3,212,749,824 params, exact match
to the analytical byte model with zero code changes): **75.9% of peak at
seq_len=512 under CUDA-graph capture**, 1.3x off the analytical floor -- a genuine
bandwidth-bound regime. `OUTLINE.md` section 2 updated to make it primary; the 1B
is kept as an architecture reference for the sections written against its specific
numbers (section 5's LM-head case study, section 7's decode budget).

Efficiency decays with context (75.9% -> 61.7% -> 50.0% at seq 512/2048/4096) because
the KV-read path is far less efficient than the weight GEMVs -- incremental bytes
between seq 512 and 4096 cost ~35 GB/s (~7.8% of peak) even graphed, against ~340
GB/s for the weights at seq 512. Worth keeping in mind for the section 11
hard-stop-at-week-7 decision: Phase 1 (weights) is optimizing a path already near
roofline on this model; Phases 3-4 (KV cache) are optimizing one at roughly a tenth
of roofline.

Predicted INT4 speedup at 3B stays large (2.5-3.4x from the byte model) and is not
diminished by any of this -- the LM head being a smaller share of total bytes here
(12.1% vs. 20-27% on the smaller models) only shrinks the *additional* win from
quantizing the head on top of the layers, not the main layer-quantization effect.
The real caveat is sequencing, not magnitude: that speedup is only visible in
wall-clock time under low-dispatch conditions (graphs/fusion) -- in plain eager mode
today, dispatch (42.5 ms) already dwarfs the fp16 floor (15.4 ms) at 3B, so an INT4
kernel alone won't move eager wall-clock much until Phase 2/5 cut dispatch down too.

### Environment fix — venv moved out of OneDrive

`myenv/` was sitting inside the OneDrive-synced project tree and had grown to
3.2 GB (`torch` alone is 2.7 GB, mostly bundled CUDA runtime libraries).
OneDrive was syncing every package file to the cloud as if it were project
content, which (a) was filling the OneDrive quota and (b) caused the
intermittent "Access is denied" errors during `uv`/`pip` installs earlier in
Phase 0 -- OneDrive holds a sync lock on files mid-write.

Moved to `C:\Users\krish\envs\LLM-testing\` (same drive, so the move was an
instant rename, not a 3 GB copy). Verified afterward: torch/CUDA and all four
packages (`transformers`, `accelerate`, `safetensors`, `datasets`) import
correctly from the new path with no reinstall needed. `Dequant_Project/`
shrank from 3.3 GB to 129 MB in the OneDrive-synced tree.

All commands below that show `../myenv/Scripts/python.exe` or
`./myenv/Scripts/python.exe` predate this move -- substitute
`C:/Users/krish/envs/LLM-testing/Scripts/python.exe` (or an equivalent
relative path) going forward.

### Step 0/1 — Environment

- Repo initialized (`Dequant_Project/`, local only — not pushed).
- Hardware: RTX 5060 Ti (sm_120, 36 SMs, 16 GB, ~448 GB/s theoretical) as the
  only measurement GPU. T600 (sm_75, 4 GB) kept as a compile target only —
  too small and too old to hold the primary model.
- `torch==2.12.0+cu132`, CUDA 13.2, driver 596.36, in `myenv/` (Python 3.14.5).
- Installed `transformers`, `accelerate`, `safetensors`, `datasets` via `uv`.
  **Gotcha**: an unconstrained `uv pip install` silently downgraded torch to
  a CPU-only PyPI build (torch's `+cu132` tag only exists on
  `download.pytorch.org`, not PyPI, so the resolver "upgraded" past it).
  Fixed and verified back to `2.12.0+cu132` with CUDA available. Any future
  install into `myenv` must pin torch/torchvision via `--constraint` and pass
  `--extra-index-url https://download.pytorch.org/whl/cu132`, or this repeats.
- Llama-3.2-1B-Instruct is gated (401, no `HF_TOKEN` yet). GPT-2 and
  Qwen2.5-1.5B-Instruct are reachable and used to verify code paths in the
  meantime.

### Step 2 — Analytical byte model

- `dequant/byte_model.py`: `spec_from_hf_config` → `param_counts` /
  `decode_bytes` / `predicted_latency_s`. Predicts bytes per decode step
  before any kernel exists, split into layer weights, LM head, KV cache,
  activations — deliberately written before measuring anything.
- `scripts/verify_byte_model.py`: fetches safetensors *headers only* via HTTP
  range requests (a few KB, not gigabytes) and checks `param_counts()`
  against the real checkpoint component-by-component.
  **Passed exactly**: gpt2 → 124,439,808 params, Qwen2.5-1.5B →
  1,543,714,304 params. Caught a real trap along the way — GPT-2's
  `h.*.attn.bias` is a causal-mask buffer, not a parameter; naively summing
  all tensor shapes would have overcounted by ~12.6M.
- `scripts/decode_bytes_report.py`: emits the Phase 0 headline numbers.
  Llama-3.2-1B is gated, so its config is hardcoded from published values
  and explicitly flagged unverified until `HF_TOKEN` is set and
  `verify_byte_model.py` confirms it against the real checkpoint.

**Headline result (GPT-2, seq_len=1024, bandwidth-bound floor
@ 448 GB/s):**

| Config | Total/token | LM head share | vs fp16 |
|---|---|---|---|
| fp16 everything | 285.7 MB | 27.0% | 1.00x |
| INT4 layers, fp16 head | 159.7 MB | 48.4% | 1.79x |
| INT4 layers, INT8 head | 121.7 MB | 32.2% | 2.35x |
| INT4 layers, INT4 head | 102.4 MB | 19.4% | 2.79x |

Confirms OUTLINE.md §5's Amdahl-law claim before any kernel is written:
leaving the LM head in fp16 caps end-to-end speedup at 1.79x regardless of
how good the layer-quantization kernel is.

Fixed a unit bug mid-step: MB was computed as `/1024**2` (MiB) while
bandwidth is quoted in decimal GB/s — switched everything to decimal MB
(`/1e6`) to keep the convention consistent before it caused a false ~5%
discrepancy later in Phase 0 Step 5 (bandwidth measurement vs. prediction).

**Status**: gate passed, committed locally (not pushed — commit
`0f8826e`). Not yet done: Steps 3–7 (fp16 baseline + golden reference,
latency sweep, achieved-bandwidth measurement, launch-overhead breakdown,
final roofline writeup).
