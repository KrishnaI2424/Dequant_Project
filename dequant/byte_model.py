"""Analytical model of bytes moved per decode step.

Phase 0, step 2. Written before any measurement so it cannot be fitted to the
data after the fact. Everything downstream is compared against this.

Two separate questions, deliberately kept apart:

  param_counts()  -- how many parameters exist. Verifiable exactly against a
                     checkpoint. Tied embeddings are counted ONCE.
  decode_bytes()  -- how many bytes a single decode step reads. A tied LM head
                     is counted here even though it is not a distinct
                     parameter, because the GEMV reads all of it every token.

That second distinction is the point of OUTLINE.md section 5.
"""

from dataclasses import dataclass

# Storage cost of one element, in bits.
BITS = {"fp32": 32, "bf16": 16, "fp16": 16, "fp8": 8, "int8": 8, "int4": 4, "int2": 2}


@dataclass(frozen=True)
class ModelSpec:
    """Architecture, reduced to the numbers that drive byte counts."""

    name: str
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    tied_embedding: bool
    gated_mlp: bool      # llama: gate+up+down. gpt2: c_fc+c_proj.
    qkv_bias: bool       # qwen2 has q/k/v biases; llama does not.
    proj_bias: bool      # output/down projection biases (gpt2 only).
    norm_bias: bool      # LayerNorm (weight+bias) vs RMSNorm (weight only).
    n_positions: int     # learned position embeddings; 0 when RoPE.

    @property
    def q_dim(self):
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self):
        return self.n_kv_heads * self.head_dim


def spec_from_hf_config(cfg, name=None):
    """Build a ModelSpec from a HuggingFace config dict."""
    mt = cfg.get("model_type")
    name = name or mt

    if mt == "gpt2":
        hidden = cfg["n_embd"]
        return ModelSpec(
            name=name,
            n_layers=cfg["n_layer"],
            hidden=hidden,
            n_heads=cfg["n_head"],
            n_kv_heads=cfg["n_head"],          # MHA
            head_dim=hidden // cfg["n_head"],
            intermediate=cfg.get("n_inner") or 4 * hidden,
            vocab=cfg["vocab_size"],
            tied_embedding=cfg.get("tie_word_embeddings", True),
            gated_mlp=False,
            qkv_bias=True,
            proj_bias=True,
            norm_bias=True,
            n_positions=cfg["n_positions"],
        )

    if mt in ("llama", "qwen2", "mistral"):
        hidden = cfg["hidden_size"]
        n_heads = cfg["num_attention_heads"]
        return ModelSpec(
            name=name,
            n_layers=cfg["num_hidden_layers"],
            hidden=hidden,
            n_heads=n_heads,
            n_kv_heads=cfg.get("num_key_value_heads", n_heads),
            head_dim=cfg.get("head_dim") or hidden // n_heads,
            intermediate=cfg["intermediate_size"],
            vocab=cfg["vocab_size"],
            tied_embedding=cfg.get("tie_word_embeddings", False),
            gated_mlp=True,
            # qwen2 hardcodes q/k/v bias; llama exposes attention_bias (default false).
            qkv_bias=cfg.get("attention_bias", mt == "qwen2"),
            proj_bias=False,
            norm_bias=False,
            n_positions=0,
        )

    raise ValueError(f"unsupported model_type: {mt!r}")


# Llama-3.2-1B-Instruct is gated on HuggingFace. These are the published config
# values, used so Phase 0 numbers can be computed before access is granted.
# UNVERIFIED against a real checkpoint -- re-run scripts/verify_byte_model.py
# once the gating request goes through.
LLAMA_3_2_1B_CONFIG = {
    "model_type": "llama",
    "num_hidden_layers": 16,
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "intermediate_size": 8192,
    "vocab_size": 128256,
    "tie_word_embeddings": True,
    "attention_bias": False,
}


def param_counts(s: ModelSpec):
    """Exact parameter counts per component. Must match the checkpoint."""
    norm_mult = 2 if s.norm_bias else 1

    if s.gated_mlp:
        mlp = 3 * s.hidden * s.intermediate
        mlp_bias = 0
    else:
        mlp = 2 * s.hidden * s.intermediate
        mlp_bias = (s.intermediate + s.hidden) if s.proj_bias else 0

    qkv = s.hidden * (s.q_dim + 2 * s.kv_dim)
    qkv_b = (s.q_dim + 2 * s.kv_dim) if s.qkv_bias else 0
    out = s.q_dim * s.hidden
    out_b = s.hidden if s.proj_bias else 0

    counts = {
        "attn_qkv": s.n_layers * (qkv + qkv_b),
        "attn_out": s.n_layers * (out + out_b),
        "mlp": s.n_layers * (mlp + mlp_bias),
        # per-layer: two norms. plus one final norm.
        "norms": s.n_layers * 2 * norm_mult * s.hidden + norm_mult * s.hidden,
        "embedding": s.vocab * s.hidden,
        "pos_embedding": s.n_positions * s.hidden,
        # A tied head is the SAME tensor as the embedding, so it is not a
        # separate parameter. It still costs bytes at decode; see decode_bytes.
        "lm_head": 0 if s.tied_embedding else s.vocab * s.hidden,
    }
    counts["total"] = sum(counts.values())
    return counts


def _bias_params(s: ModelSpec):
    """Bias parameters folded into the projection counts, which stay fp16."""
    per_layer = 0
    if s.qkv_bias:
        per_layer += s.q_dim + 2 * s.kv_dim
    if s.proj_bias:
        per_layer += s.hidden
        if not s.gated_mlp:
            per_layer += s.intermediate + s.hidden
    return s.n_layers * per_layer


def _quant_bytes(numel, dtype, group_size=None, asymmetric=False):
    """Bytes for numel values at dtype, including per-group scale overhead.

    Scales and zero-points are stored fp16, one pair per group. group_size=None
    means no grouping metadata (the 16-bit baseline).
    """
    body = numel * BITS[dtype] / 8
    if group_size is None or BITS[dtype] >= 16:
        return body
    n_groups = numel / group_size
    meta_per_group = 2 * (2 if asymmetric else 1)  # fp16 scale (+ fp16 zero-point)
    return body + n_groups * meta_per_group


def decode_bytes(
    s: ModelSpec,
    seq_len,
    weight_dtype="fp16",
    lm_head_dtype=None,
    kv_dtype="fp16",
    group_size=128,
    asymmetric=False,
):
    """Bytes read for ONE decode step (batch 1) at a given context length.

    Split into the terms that matter: transformer-layer weights (fixed), the
    LM head (fixed, and large on small models), and the KV cache (grows with
    seq_len).
    """
    lm_head_dtype = lm_head_dtype or weight_dtype

    # --- transformer layer weights ------------------------------------------
    # Only projection matrices get quantized. Norms and biases stay fp16: tiny,
    # and quantizing them costs accuracy for nothing (OUTLINE section 5).
    p = param_counts(s)
    biases = _bias_params(s)
    quantizable = p["attn_qkv"] + p["attn_out"] + p["mlp"] - biases
    small = p["norms"] + biases

    layer_weights = _quant_bytes(quantizable, weight_dtype, group_size, asymmetric)
    layer_weights += small * BITS["fp16"] / 8

    # --- LM head -------------------------------------------------------------
    # Read in full by the final GEMV every token, tied or not.
    lm_head = _quant_bytes(s.vocab * s.hidden, lm_head_dtype, group_size, asymmetric)

    # The input embedding lookup reads ONE row, not the whole table.
    embed_lookup = s.hidden * BITS[lm_head_dtype] / 8

    # --- KV cache ------------------------------------------------------------
    # K and V for every past token, every layer, every KV head.
    kv_cache = 2 * s.n_layers * s.kv_dim * seq_len * BITS[kv_dtype] / 8
    if BITS[kv_dtype] < 16:
        # Per-token, per-head, per-tensor fp16 scale.
        kv_cache += 2 * s.n_layers * s.n_kv_heads * seq_len * 2

    # --- activations ---------------------------------------------------------
    # Batch 1, so these are single vectors. Estimate: each kernel in the decode
    # path reads its input and writes its output. Counted explicitly rather
    # than hand-waved, but it is an estimate -- Nsight is the arbiter.
    h, i = s.hidden, s.intermediate
    qkv_out = s.q_dim + 2 * s.kv_dim
    if s.gated_mlp:
        mlp_act = h + 2 * i + 2 * i + i + i + h   # gate/up write, swiglu r/w, down
    else:
        mlp_act = h + i + i + i + h
    per_layer_act = (
        2 * h              # pre-attn norm: read x, write normed
        + h + qkv_out      # qkv projection
        + qkv_out          # rope + attention read
        + s.q_dim + h      # attn out -> o_proj
        + 2 * h            # residual
        + 2 * h            # post-attn norm
        + mlp_act
        + 2 * h            # residual
    )
    # + final norm, + logits written by the LM head GEMV.
    activations = (s.n_layers * per_layer_act + 2 * h + s.vocab) * BITS["fp16"] / 8

    out = {
        "layer_weights": layer_weights,
        "lm_head": lm_head,
        "kv_cache": kv_cache,
        "activations": activations + embed_lookup,
    }
    out["total"] = sum(out.values())
    return out


def predicted_latency_s(total_bytes, bandwidth_gb_s):
    """Bandwidth-bound floor: bytes / achieved bandwidth."""
    return total_bytes / (bandwidth_gb_s * 1e9)
