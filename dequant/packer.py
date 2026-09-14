"""Offline weight quantization and packing.

Phase 1, per OUTLINE.md section 5. Every format axis the sweep varies is a
parameter here, not a constant: bit width, group size, symmetry, packing
layout.

Conventions that the kernel MUST agree with
-------------------------------------------
Weights are [N, K] and the GEMV reduces over K, so groups run along K -- each
group of `group_size` consecutive input elements shares one scale. This
matches GPTQ/AWQ. scales are [N, K // group_size].

Codes are always stored UNSIGNED. Symmetric quantization biases by 2^(b-1)
before packing, so the kernel unpacks a plain unsigned field and subtracts the
bias (the XOR-0x8-then-subtract-8 trick in section 6 for 4-bit). Packing never
has to think about sign.

Packing layouts (4-bit, 8 values per uint32):
    sequential   value i occupies bits [4i, 4i+4). Simple, one shift+mask each.
    interleaved  nibble order [0,2,4,6,1,3,5,7]. Lets the kernel pull four
                 values at once with (w >> 0) & 0x0F0F0F0F and the other four
                 with (w >> 4) & 0x0F0F0F0F -- two ops instead of eight. This
                 is the axis section 5 says nobody documents clearly, so it is
                 measured rather than assumed.

Layout mismatch between packer and kernel is the number-one source of silent
wrong answers in this project (section 11), so QuantConfig is carried around
with the data and the kernel asserts against it.
"""

from dataclasses import dataclass

import torch

# Logical index order within a 32-bit word for the interleaved layout.
_INTERLEAVE = {
    4: [0, 2, 4, 6, 1, 3, 5, 7],
    2: [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15],
    8: [0, 2, 1, 3],
}


@dataclass(frozen=True)
class QuantConfig:
    bits: int = 4
    group_size: int | None = 128   # None means per-channel (one scale per row).
    symmetric: bool = True
    layout: str = "sequential"     # "sequential" | "interleaved"

    def __post_init__(self):
        if self.bits not in (2, 4, 8):
            raise ValueError(f"bits must be 2, 4 or 8, got {self.bits}")
        if self.layout not in ("sequential", "interleaved"):
            raise ValueError(f"unknown layout: {self.layout!r}")
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError("group_size must be positive or None")

    @property
    def per_pack(self):
        """Values packed into one uint32 word."""
        return 32 // self.bits

    @property
    def qmax(self):
        """Largest storable unsigned code."""
        return (1 << self.bits) - 1

    @property
    def bias(self):
        """Offset applied to symmetric codes so they store unsigned."""
        return 1 << (self.bits - 1)

    def groups_for(self, k):
        return 1 if self.group_size is None else k // self.group_size

    def effective_group_size(self, k):
        return k if self.group_size is None else self.group_size


@dataclass
class PackedWeight:
    """A quantized weight plus everything needed to reconstruct it."""

    qweight: torch.Tensor          # uint32 [N, K // per_pack]
    scales: torch.Tensor           # fp16   [N, n_groups]
    zeros: torch.Tensor | None     # fp16   [N, n_groups], asymmetric only
    config: QuantConfig
    shape: tuple                   # original [N, K]

    @property
    def nbytes(self):
        n = self.qweight.numel() * 4 + self.scales.numel() * 2
        if self.zeros is not None:
            n += self.zeros.numel() * 2
        return n


def quantize(w, cfg: QuantConfig):
    """Quantize [N, K] -> unsigned integer codes, scales, zeros.

    Returns codes as int32 in [0, qmax]. Symmetric codes are already biased.
    """
    if w.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(w.shape)}")
    n, k = w.shape
    gs = cfg.effective_group_size(k)
    if k % gs:
        raise ValueError(f"K={k} is not divisible by group_size={gs}")

    wf = w.float().reshape(n, k // gs, gs)

    if cfg.symmetric:
        # scale = max|w| / (2^(b-1) - 1); no zero-point, no subtract in the
        # kernel's inner loop.
        amax = wf.abs().amax(dim=-1, keepdim=True)
        scales = amax / (cfg.bias - 1)
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        # Round to fp16 BEFORE deriving codes. Scales are stored fp16, so the
        # kernel reconstructs with the fp16 value; choosing codes against the
        # fp32 one makes them subtly wrong. At 8-bit the codes reach +-127, so
        # the scale's own fp16 rounding gets multiplied by 127 and eats ~7% of
        # the half-step error budget. This is the packer/kernel disagreement
        # section 11 warns about, in miniature.
        scales = scales.to(torch.float16).float()
        q = torch.round(wf / scales) + cfg.bias
        zeros = None
    else:
        # scale = (max - min) / (2^b - 1); costs a stored zero-point per group
        # and an extra subtract per element. Section 5 wants that cost measured.
        wmax = wf.amax(dim=-1, keepdim=True)
        wmin = wf.amin(dim=-1, keepdim=True)
        scales = (wmax - wmin) / cfg.qmax
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        scales = scales.to(torch.float16).float()   # same reasoning as above
        zeros = torch.round(-wmin / scales)
        q = torch.round(wf / scales) + zeros

    q = q.clamp_(0, cfg.qmax).to(torch.int32).reshape(n, k)
    scales = scales.reshape(n, -1).to(torch.float16)
    if zeros is not None:
        zeros = zeros.reshape(n, -1).to(torch.float16)
    return q, scales, zeros


def pack(q, cfg: QuantConfig):
    """Pack unsigned codes [N, K] into uint32 words [N, K // per_pack]."""
    n, k = q.shape
    per = cfg.per_pack
    if k % per:
        raise ValueError(f"K={k} is not divisible by {per} values per word")

    order = _INTERLEAVE[cfg.bits] if cfg.layout == "interleaved" else range(per)
    blocks = q.reshape(n, k // per, per)

    out = torch.zeros(n, k // per, dtype=torch.int64, device=q.device)
    for slot, logical in enumerate(order):
        out |= (blocks[:, :, logical].to(torch.int64) & cfg.qmax) << (cfg.bits * slot)
    # int64 -> uint32 bit pattern. torch has no uint32 arithmetic, so the
    # kernel reads this as unsigned and Python keeps it in int32's bit pattern.
    return out.to(torch.int32)


def unpack(packed, cfg: QuantConfig, k):
    """Inverse of pack(). Returns unsigned codes [N, K] as int32."""
    n = packed.shape[0]
    per = cfg.per_pack
    order = _INTERLEAVE[cfg.bits] if cfg.layout == "interleaved" else range(per)

    words = packed.to(torch.int64) & 0xFFFFFFFF
    out = torch.zeros(n, k // per, per, dtype=torch.int32, device=packed.device)
    for slot, logical in enumerate(order):
        out[:, :, logical] = ((words >> (cfg.bits * slot)) & cfg.qmax).to(torch.int32)
    return out.reshape(n, k)


def dequantize(pw: PackedWeight):
    """Reconstruct the fp16 weight from a PackedWeight."""
    cfg = pw.config
    n, k = pw.shape
    gs = cfg.effective_group_size(k)

    q = unpack(pw.qweight, cfg, k).reshape(n, k // gs, gs).float()
    scales = pw.scales.float().unsqueeze(-1)
    if cfg.symmetric:
        w = (q - cfg.bias) * scales
    else:
        w = (q - pw.zeros.float().unsqueeze(-1)) * scales
    return w.reshape(n, k).to(torch.float16)


def quantize_tensor(w, cfg: QuantConfig) -> PackedWeight:
    """Quantize and pack in one step."""
    q, scales, zeros = quantize(w, cfg)
    return PackedWeight(
        qweight=pack(q, cfg),
        scales=scales,
        zeros=zeros,
        config=cfg,
        shape=tuple(w.shape),
    )
