# Progress Log

Running log of what's actually been done, in case a session gets cut short.
Newest entries at the top. See `OUTLINE.md` for the full plan.

---

## Phase 0 — Instrumentation and roofline

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
