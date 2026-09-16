"""The quantized weight as a drop-in nn.Linear replacement.

Phase 1. `dequant/packer.py` turns an fp16 [N, K] weight into packed int32
words plus fp16 scales (and zeros); this module is what the model actually
calls at decode time. The packed form is the ONLY copy of the weight that
exists in VRAM after `quantize_model()` runs -- there is no `.weight`
attribute here on purpose, so nothing can silently fall back to a full-size
fp16 matmul.

Two paths, chosen by M = the number of rows in the activation:

    M <= MAX_M   the dequant-GEMV kernel. Weights are unpacked in registers
                 and discarded; ~4.25 bits/weight crosses DRAM instead of 16.
                 This is the decode path and the point of the project.
    M >  MAX_M   dequantize to a temporary fp16 matrix and call F.linear.
                 Prefill is compute-bound, so materializing costs little and
                 keeps the model correct end to end (OUTLINE section 7).

`forward` never touches the CPU and never synchronizes, so a decode step
through these modules is CUDA-graph capturable (which is the only regime
where this GPU's real latency is observable -- see dequant/bench.py).
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from dequant.packer import PackedWeight, QuantConfig, quantize_tensor

# The .pyd is built --inplace next to setup.py, i.e. at the project root, and
# dequant/ is not a package on disk with an __init__.py, so the root is what
# has to go on sys.path.
_ROOT = Path(__file__).resolve().parent.parent

# Largest M the kernel's MT=8 instantiation handles. Above this, fall back.
MAX_M = 8

# The seven projections per decoder layer that this project quantizes. Same
# tuple as scripts/eval_quantized.py; norms, embeddings and the LM head are
# deliberately left fp16 (OUTLINE section 5).
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

_ext = None


def _get_ext():
    """Import the prebuilt extension module (dequant_cuda.*.pyd) at the root."""
    global _ext
    if _ext is not None:
        return _ext

    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    try:
        import dequant_cuda
    except ImportError as exc:
        raise ImportError(
            "dequant_cuda extension is not built. Build it once with:\n"
            f"    cd {_ROOT}\n"
            "    python setup.py build_ext --inplace\n"
            "Re-run that after any change to kernels/."
        ) from exc

    _ext = dequant_cuda
    return _ext


def gemv(x, pw: PackedWeight):
    """y = x @ W^T with W held packed. x is [..., K] fp16, y is [..., N] fp16."""
    n, k = pw.shape
    cfg = pw.config
    x2 = x.reshape(-1, k).contiguous()
    y = _get_ext().gemv(
        x2, pw.qweight, pw.scales, pw.zeros,
        cfg.bits, cfg.effective_group_size(k), cfg.symmetric,
        cfg.layout == "interleaved",
    )
    return y.view(*x.shape[:-1], n)


def dequantize_cuda(pw: PackedWeight):
    """Reconstruct the fp16 [N, K] weight on the GPU. Bit-exact with the packer."""
    n, k = pw.shape
    cfg = pw.config
    return _get_ext().dequantize(
        pw.qweight, pw.scales, pw.zeros,
        cfg.bits, cfg.effective_group_size(k), cfg.symmetric,
        cfg.layout == "interleaved", k,
    )


class QuantLinear(nn.Module):
    """nn.Linear whose weight lives only in packed form."""

    def __init__(self, pw: PackedWeight, bias=None):
        super().__init__()
        self.out_features, self.in_features = pw.shape
        self.config = pw.config
        self.register_buffer("qweight", pw.qweight)
        self.register_buffer("scales", pw.scales)
        self.register_buffer("zeros", pw.zeros)
        self.register_buffer("bias", bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, cfg: QuantConfig):
        pw = quantize_tensor(linear.weight.data, cfg)
        bias = None if linear.bias is None else linear.bias.data.to(torch.float16)
        return cls(pw, bias)

    def _packed(self):
        # Rebuilt per call rather than cached so that .to(), .cuda() and any
        # other buffer move stay correct -- a cached PackedWeight would keep
        # pointing at the tensors from whichever device it was built on.
        return PackedWeight(
            qweight=self.qweight,
            scales=self.scales,
            zeros=self.zeros,
            config=self.config,
            shape=(self.out_features, self.in_features),
        )

    def forward(self, x):
        pw = self._packed()
        m = x.numel() // self.in_features
        if m <= MAX_M:
            y = gemv(x, pw)
        else:
            y = F.linear(x, dequantize_cuda(pw))
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self):
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}, config={self.config}")


def quantize_model(model, cfg: QuantConfig, names=PROJ):
    """Replace every nn.Linear named in `names` with a QuantLinear, in place."""
    replaced = 0
    packed_bytes = 0
    fp16_bytes = 0

    for path, mod in list(model.named_modules()):
        leaf = path.rsplit(".", 1)[-1]
        if leaf not in names or not isinstance(mod, nn.Linear):
            continue
        parent = model.get_submodule(path.rsplit(".", 1)[0] if "." in path else "")
        ql = QuantLinear.from_linear(mod, cfg)
        setattr(parent, leaf, ql)
        replaced += 1
        packed_bytes += ql._packed().nbytes
        fp16_bytes += mod.weight.numel() * 2

    torch.cuda.empty_cache()
    return {"replaced": replaced, "packed_bytes": packed_bytes,
            "fp16_bytes": fp16_bytes}
