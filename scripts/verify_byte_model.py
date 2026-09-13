"""Verify byte_model.param_counts() against real checkpoints.

The Phase 0 gate: if the analytical parameter count does not match the
checkpoint exactly, every downstream byte number is wrong.

Reads safetensors headers over HTTP range requests -- a few KB per model
instead of gigabytes of weights. The header lists every tensor name, dtype
and shape, which is all a parameter count needs.

    python scripts/verify_byte_model.py
"""

import json
import struct
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dequant.byte_model import param_counts, spec_from_hf_config

# Tensors present in a checkpoint that are NOT nn.Parameters. GPT-2 ships its
# causal mask as a buffer; counting it would add 12 x 1024 x 1024 phantom
# "parameters". This is exactly the trap this script exists to catch.
BUFFER_SUFFIXES = (".attn.bias", ".attn.masked_bias", ".rotary_emb.inv_freq")

MODELS = ["gpt2", "Qwen/Qwen2.5-1.5B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"]


def _get(url, headers=None):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=headers or {}), timeout=60
    ).read()


def safetensors_header(repo, filename):
    """Fetch just the JSON header of a remote safetensors file."""
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    n = struct.unpack("<Q", _get(url, {"Range": "bytes=0-7"})[:8])[0]
    raw = _get(url, {"Range": f"bytes=8-{8 + n - 1}"})
    return json.loads(raw[:n])


def checkpoint_tensors(repo):
    """All {name: shape} in a checkpoint, sharded or not."""
    try:
        index = json.loads(
            _get(f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json")
        )
        files = sorted(set(index["weight_map"].values()))
    except Exception:
        files = ["model.safetensors"]

    tensors = {}
    for f in files:
        for name, meta in safetensors_header(repo, f).items():
            if name != "__metadata__":
                tensors[name] = meta["shape"]
    return tensors


def classify(name):
    """Map a checkpoint tensor name to a param_counts() component."""
    if name.endswith(BUFFER_SUFFIXES):
        return None  # buffer, not a parameter
    n = name.replace("transformer.", "").replace("model.", "")

    if n in ("wte.weight", "embed_tokens.weight"):
        return "embedding"
    if n == "wpe.weight":
        return "pos_embedding"
    if n.startswith("lm_head"):
        return "lm_head"
    if "ln_f" in n or n.startswith("norm."):
        return "norms"
    if "layernorm" in n or "ln_1" in n or "ln_2" in n:
        return "norms"
    if "c_attn" in n or any(k in n for k in ("q_proj", "k_proj", "v_proj")):
        return "attn_qkv"
    if "o_proj" in n or "attn.c_proj" in n:
        return "attn_out"
    if "mlp" in n:
        return "mlp"
    raise ValueError(f"unclassified tensor: {name}")


def numel(shape):
    out = 1
    for d in shape:
        out *= d
    return out


def verify(repo):
    print(f"\n=== {repo}")
    try:
        cfg = json.loads(_get(f"https://huggingface.co/{repo}/resolve/main/config.json"))
    except Exception as e:
        code = getattr(e, "code", None)
        if code == 401:
            print("  SKIP: gated repo, no access (401). Request access + set HF_TOKEN.")
        else:
            print(f"  SKIP: config unreachable ({type(e).__name__})")
        return None

    spec = spec_from_hf_config(cfg, name=repo)
    predicted = param_counts(spec)

    actual = {}
    for name, shape in checkpoint_tensors(repo).items():
        comp = classify(name)
        if comp is not None:
            actual[comp] = actual.get(comp, 0) + numel(shape)
    actual["total"] = sum(actual.values())

    ok = True
    print(f"  {'component':<16}{'predicted':>14}{'actual':>14}   delta")
    for comp in ("attn_qkv", "attn_out", "mlp", "norms", "embedding",
                 "pos_embedding", "lm_head", "total"):
        p, a = predicted.get(comp, 0), actual.get(comp, 0)
        mark = "" if p == a else f"  <-- MISMATCH {a - p:+d}"
        if p != a:
            ok = False
        print(f"  {comp:<16}{p:>14,}{a:>14,}{mark}")

    print(f"  tied_embedding={spec.tied_embedding}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = {m: verify(m) for m in MODELS}
    checked = {m: r for m, r in results.items() if r is not None}
    print(f"\n{sum(checked.values())}/{len(checked)} verified, "
          f"{len(results) - len(checked)} skipped")
    sys.exit(0 if all(checked.values()) else 1)
