# Progress Log

Running log of what's actually been done, in case a session gets cut short.
Newest entries at the top. See `OUTLINE.md` for the full plan.

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
