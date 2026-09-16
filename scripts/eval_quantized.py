"""Accuracy reference: what does quantizing the real model actually cost?

Phase 1, the accuracy half of the format sweep. The dequant-GEMV kernel does
not exist yet, so this uses SIMULATED quantization: real weights are quantized
with dequant.packer and immediately dequantized back to fp16 in place. The
model then moves exactly as many bytes as fp16 did, so it isolates the
question the kernel cannot answer anyway -- how much quality each format
costs -- independent of speed.

Decode timing IS also measured here (CUDA-graph capture, via dequant.bench,
the near-roofline regime), but it is expected to be FLAT across formats and
that flatness is the correct result, not a null finding: since every format
is ordinary fp16 by the time the forward pass runs, there is no mechanism by
which a "format" label could change decode latency yet. It is recorded to
make that flatness explicit and measured rather than assumed, and as the
fp16 reference point the real kernel's eventual numbers get compared against.

Only the seven projection types are quantized (q/k/v/o/gate/up/down), matching
OUTLINE section 5. Norms stay fp16, and the LM head is left alone so its
separate treatment (section 5) stays a separate experiment.

    python scripts/eval_quantized.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dequant.bench import time_decode_step
from dequant.byte_model import decode_bytes, spec_from_hf_config
from dequant.packer import QuantConfig, dequantize, quantize_tensor

MODEL = "unsloth/Llama-3.2-3B-Instruct"
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# Measured device-to-device copy bandwidth on this card (see PROGRESS.md,
# Phase 1 toolchain entry) -- the honest denominator for "% of peak", not the
# 448 GB/s theoretical figure nothing can actually reach.
PRACTICAL_PEAK_GB_S = 384.0
DECODE_SEQ_LEN = 512  # matches the seq_len already reported elsewhere for this model

# Formats worth a reference point. One axis varied at a time from the
# INT4 g128 symmetric baseline, per OUTLINE section 11's "sweep one axis at a
# time" guidance.
FORMATS = [
    ("fp16 (baseline)", None),
    ("INT8 g128 sym", QuantConfig(8, 128, True)),
    ("INT4 g32  sym", QuantConfig(4, 32, True)),
    ("INT4 g128 sym", QuantConfig(4, 128, True)),
    ("INT4 g128 asym", QuantConfig(4, 128, False)),
    ("INT4 per-chan sym", QuantConfig(4, None, True)),
    ("INT2 g128 sym", QuantConfig(2, 128, True)),
]

PROMPT = "The capital of France is"


def target_linears(model):
    for layer in model.model.layers:
        for name in PROJ:
            mod = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            if mod is not None:
                yield name, mod


def apply_quantization(model, originals, cfg):
    """Restore fp16 weights, then fake-quantize in place. Returns error stats."""
    abs_err = 0.0
    abs_mag = 0.0
    n = 0
    for (name, mod), orig in zip(target_linears(model), originals):
        w = orig.to(mod.weight.device)
        if cfg is None:
            mod.weight.data.copy_(w)
            continue
        deq = dequantize(quantize_tensor(w, cfg))
        abs_err += (deq.float() - w.float()).abs().sum().item()
        abs_mag += w.float().abs().sum().item()
        n += w.numel()
        mod.weight.data.copy_(deq)
    return {"mean_abs_err": abs_err / n if n else 0.0,
            "rel_err": abs_err / abs_mag if abs_mag else 0.0}


@torch.no_grad()
def perplexity(model, ids, window=2048):
    """Perplexity over a fixed token sequence, non-overlapping windows."""
    nll, count = 0.0, 0
    for i in range(0, ids.numel() - 1, window):
        chunk = ids[:, i:i + window + 1]
        if chunk.numel() < 2:
            break
        out = model(input_ids=chunk[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            out.logits[0].float(), chunk[0, 1:], reduction="sum")
        nll += loss.item()
        count += chunk.numel() - 1
    return float(torch.exp(torch.tensor(nll / count)))


def load_eval_text(tok, target_tokens):
    """Wikitext-2 test split.

    The dataset lives under the Salesforce org; the bare "wikitext" path fails
    on datasets 5.x with HfUriError. There is deliberately no synthetic
    fallback: repetitive filler text gives a baseline perplexity near 1.0,
    where the model has no uncertainty left to lose and quantization damage
    cannot register. A fallback like that does not degrade the measurement,
    it silently invalidates it -- better to fail and say so.
    """
    from datasets import load_dataset
    last = None
    for repo in ("Salesforce/wikitext", "wikitext"):
        try:
            ds = load_dataset(repo, "wikitext-2-raw-v1", split="test")
            text = "\n\n".join(t for t in ds["text"] if t.strip())
            ids = tok(text, return_tensors="pt").input_ids[:, :target_tokens]
            return ids, f"{repo} wikitext-2-raw-v1 test"
        except Exception as e:  # noqa: PERF203 - want the last error reported
            last = e
            print(f"  ({repo} unavailable: {type(e).__name__})")
    raise RuntimeError(
        "Could not load wikitext-2. Perplexity on substitute filler text is "
        f"meaningless, so refusing to report a number. Last error: {last!r}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tokens", type=int, default=8192,
                    help="tokens of eval text for perplexity")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model = model.to("cuda").eval()

    # Keep pristine fp16 copies on CPU; each format restores from these so
    # errors never compound across configs.
    originals = [m.weight.data.detach().to("cpu", copy=True)
                 for _, m in target_linears(model)]
    quantized_params = sum(o.numel() for o in originals)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}")
    print(f"  quantizing {quantized_params:,} of {total_params:,} params "
          f"({quantized_params / total_params * 100:.1f}%) -- "
          f"LM head and norms left fp16")

    ids, source = load_eval_text(tok, args.tokens)
    ids = ids.to("cuda")
    print(f"  perplexity over {ids.numel():,} tokens of {source}")

    # Real bytes moved during decode are ALWAYS the fp16 count here: fake
    # quantization dequantizes back to fp16 before the forward pass runs, so
    # every format below moves identical bytes regardless of its label. This
    # is the denominator for "% of peak" in the decode table -- it is not the
    # bytes a real INT4 kernel would move (see decode_bytes_report.py / Phase
    # 1 status for that hypothetical).
    spec = spec_from_hf_config(model.config.to_dict(), name=args.model)
    real_decode_bytes = decode_bytes(spec, DECODE_SEQ_LEN,
                                     weight_dtype="fp16", lm_head_dtype="fp16")["total"]
    print(f"  decode timing at seq_len={DECODE_SEQ_LEN}, CUDA-graph capture "
          f"(the only near-roofline regime -- eager is dispatch-bound, see "
          f"PROGRESS.md Phase 0)\n")

    prompt_ids = tok(PROMPT, return_tensors="pt").to("cuda")
    results = []
    print(f"  {'format':<20}{'rel err':>9}{'ppl':>9}{'dppl':>8}   generation")
    print(f"  {'':<20}{'decode ms':>11}{'tok/s':>9}{'%peak':>8}   "
          f"(fake-quant -- expect flat; real bytes moved are always fp16)")
    base_ppl = None
    for label, cfg in FORMATS:
        stats = apply_quantization(model, originals, cfg)
        t0 = time.time()
        ppl = perplexity(model, ids)
        with torch.no_grad():
            gen = model.generate(**prompt_ids, max_new_tokens=8, do_sample=False)
        text = tok.decode(gen[0][prompt_ids.input_ids.shape[1]:]).strip()
        text = text.replace("\n", " ")[:38]

        decode = time_decode_step(model, DECODE_SEQ_LEN, cache_impl="graph",
                                  warmup=10, iters=30)
        decode_ms = decode["total_ms"]["median"]
        tok_s = 1e3 / decode_ms
        achieved_gb_s = real_decode_bytes / (decode_ms * 1e-3) / 1e9
        pct_peak = achieved_gb_s / PRACTICAL_PEAK_GB_S * 100

        if base_ppl is None:
            base_ppl = ppl
        results.append({
            "format": label, "ppl": ppl, "delta_ppl": ppl - base_ppl,
            "eval_s": time.time() - t0,
            "decode_ms": decode_ms, "decode_tok_s": tok_s,
            "decode_achieved_gb_s": achieved_gb_s, "decode_pct_peak": pct_peak,
            **stats,
        })
        print(f"  {label:<20}{stats['rel_err'] * 100:>8.2f}%{ppl:>9.3f}"
              f"{ppl - base_ppl:>+8.3f}   {text!r}")
        print(f"  {'':<20}{decode_ms:>11.3f}{tok_s:>9.0f}{pct_peak:>7.1f}%")

    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out / f"quant_accuracy_{args.model.replace('/', '_')}_{stamp}.json"
    path.write_text(json.dumps({
        "model": args.model, "eval_source": source,
        "eval_tokens": int(ids.numel()),
        "quantized_params": quantized_params, "total_params": total_params,
        "results": results,
    }, indent=2))
    print(f"\nwrote {path.relative_to(path.parent.parent)}")


if __name__ == "__main__":
    main()
