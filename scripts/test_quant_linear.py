"""Verification gate for the Python layer: gemv(), QuantLinear, quantize_model.

scripts/test_dequant_kernel.py gates the kernel itself. This gates everything
between the kernel and the model -- the reshapes, the M > 8 fallback, the
buffer registration, the module surgery -- since a bug there produces wrong
answers that look exactly like a wrong kernel.

The accuracy criterion is the same one used for the kernel, and it is a
statement about floating point, not a fudge factor:

    |y - ref| <= 2^-11 * |ref| + 2^-20 * (|W_deq| @ |x|)

ref is the fp32 product of the SAME dequantized weights, so quantization
error cancels out entirely and only arithmetic rounding is left. The first
term is one fp16 ulp on the output; the second bounds the fp32 accumulation
over K terms by the magnitude that actually flowed through the sum, which is
what catches a reduction done in the wrong order or the wrong precision.
Comparing against the original fp16 weight instead would drown that signal in
the quantization error the format is supposed to have.

WHICH dequantized weight, though, is not the same for both paths, and getting
it wrong makes the test measure the reference instead of the code:

    M <= 8, gemv          the kernel holds (code - off) * scale in an fp32
                          register and multiplies it straight into the
                          accumulator. It never materializes an fp16 weight,
                          so it is strictly MORE precise than
                          packer.dequantize(), whose fp16 store costs a 2^-11
                          rounding per element. Charging the kernel for that
                          rounding puts the reference's own error ~14x over
                          this budget. Reference: exact_weight() below.
    M >  8, fallback      QuantLinear really does call dequantize_cuda() and
                          hand an fp16 [N, K] matrix to F.linear, so the fp16
                          weight IS what was computed and packer.dequantize()
                          is the honest reference. Using exact_weight() here
                          fails at ~7x, correctly.

    python scripts/test_quant_linear.py
"""

import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dequant import packer
from dequant.packer import QuantConfig, quantize_tensor
from dequant.quant_linear import (
    QuantLinear,
    dequantize_cuda,
    gemv,
    quantize_model,
)

# The reference must be plain fp32, not tensor-float32, or the "reference" is
# itself only ~10 bits of mantissa and the criterion means nothing.
torch.backends.cuda.matmul.allow_tf32 = False

# Same class of fix, for the thing being measured rather than the reference.
# This defaults to True, and it lets cuBLAS round a split-K matmul's PARTIAL
# sums to fp16 before combining them. The criterion's second term is an fp32
# accumulation budget (2^-20), so an fp16 partial-sum rounding (2^-11 of a
# partial that can be far larger than the cancelled result) blows straight
# through it: measured 20.9x over at M=64, and 0.95x with this off, on pure
# torch with no kernel of ours involved. M=9 happens to pick a non-split-K
# path and never showed it. This is ordinary PyTorch behaviour and not a bug
# in the fallback -- but it is not the arithmetic the criterion describes, so
# it is switched off here rather than papered over with a bigger tolerance.
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def criterion(y, ref, a, depth=1):
    """Elementwise pass/fail plus the worst observed ratio, for reporting."""
    tol = depth * (2 ** -11 * ref.abs() + 2 ** -20 * a)
    ratio = ((y.float() - ref).abs() / (tol + 1e-30)).max().item()
    return ratio <= 1.0, f"worst={ratio:.3f}x tol"


def exact_weight(pw):
    """(code - offset) * scale in fp32, WITHOUT packer.dequantize's fp16 store.

    Same helper as scripts/test_dequant_kernel.py, and the same reason: this is
    what the GEMV kernel actually multiplies by, so it is the reference that
    isolates the dot product instead of measuring the reference's own rounding.
    See the module docstring for which path takes which reference.
    """
    cfg = pw.config
    n, k = pw.shape
    gs = cfg.effective_group_size(k)
    q = packer.unpack(pw.qweight, cfg, k).reshape(n, k // gs, gs).float()
    off = cfg.bias if cfg.symmetric else pw.zeros.float().unsqueeze(-1)
    return ((q - off) * pw.scales.float().unsqueeze(-1)).reshape(n, k)


def test_gemv_reshape():
    print("\nA. gemv() on a 3-D input (M=6 after flatten)")
    torch.manual_seed(0)
    n, k = 3072, 3072
    w = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.02
    pw = quantize_tensor(w, QuantConfig(4, 128, False))
    x = torch.randn(2, 3, k, dtype=torch.float16, device="cuda")

    y = gemv(x, pw)
    check("shape [2, 3, N]", tuple(y.shape) == (2, 3, n), str(tuple(y.shape)))

    we = exact_weight(pw)          # M=6 -> gemv path
    xf = x.float()
    ref = F.linear(xf, we)
    a = F.linear(xf.abs(), we.abs())
    check("INT4 g128 asym [3072,3072]", *criterion(y, ref, a))


def test_quant_linear():
    print("\nB. QuantLinear vs F.linear, including the M > 8 fallback")
    torch.manual_seed(0)
    k, n = 3072, 1024
    lin = nn.Linear(k, n, bias=True).to("cuda", torch.float16)
    cfg = QuantConfig(4, 128, False)
    ql = QuantLinear.from_linear(lin, cfg)
    pw = ql._packed()

    check("dequantize_cuda is bit-exact with the packer",
          torch.equal(dequantize_cuda(pw), packer.dequantize(pw)))

    deq = packer.dequantize(pw).float()
    we = exact_weight(pw)
    bias = ql.bias.float()
    for shape in [(1, k), (8, k), (9, k), (64, k), (2, 32, k)]:
        x = torch.randn(*shape, dtype=torch.float16, device="cuda")
        y = ql(x)
        xf = x.float()
        m = x.numel() // k
        # The reference has to be the weight the path in question actually
        # used: unrounded fp32 for the register-only kernel, the fp16 matrix
        # for the fallback that really does materialize one. See the module
        # docstring.
        gemv_path = m <= 8
        w_ref = we if gemv_path else deq
        ref = F.linear(xf, w_ref, bias)
        a = F.linear(xf.abs(), w_ref.abs())
        path = "gemv" if gemv_path else "dequant+F.linear"
        # depth 2: QuantLinear rounds to fp16 twice on this path -- once when
        # the kernel writes y, once for the separate bias add -- while the
        # reference fuses the bias and rounds once.
        ok, detail = criterion(y, ref, a, depth=2)
        check(f"{tuple(shape)} M={m} via {path}", ok, detail)


class _Attn(nn.Module):
    def __init__(self, h):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(h, h, bias=False))


class _Mlp(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)


class _Block(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.self_attn = _Attn(h)
        self.mlp = _Mlp(h, i)

    def forward(self, x):
        a = self.self_attn
        x = a.o_proj(a.q_proj(x) + a.k_proj(x) + a.v_proj(x))
        return self.mlp.down_proj(self.mlp.gate_proj(x) + self.mlp.up_proj(x))


class _Fake(nn.Module):
    """Llama's module names, no nonlinearity.

    Purely linear on purpose: it lets the same forward be run with absolute
    weights on absolute inputs, which gives a genuine elementwise bound on the
    magnitudes flowing through the network -- the |W| @ |x| term the criterion
    needs, but for a whole stack rather than one matmul.
    """

    def __init__(self, h=256, i=512, n_blocks=2):
        super().__init__()
        self.norm = nn.LayerNorm(h)
        self.layers = nn.ModuleList(_Block(h, i) for _ in range(n_blocks))
        self.lm_head = nn.Linear(h, h, bias=False)

    def body(self, h):
        for blk in self.layers:
            h = blk(h)
        return self.lm_head(h)

    def forward(self, x):
        return self.body(self.norm(x))


def test_quantize_model():
    print("\nC. quantize_model on a fake Llama-shaped model")
    torch.manual_seed(0)
    h = 256
    model = _Fake(h).to("cuda", torch.float16).eval()

    # fp16 reference: the same weights, dequantized back in place, so the only
    # difference from the quantized model is which kernel does the matmul.
    ref_model = copy.deepcopy(model)
    cfg = QuantConfig(4, 128, False)
    for mod in ref_model.modules():
        if isinstance(mod, nn.Linear) and mod is not ref_model.lm_head:
            mod.weight.data.copy_(packer.dequantize(quantize_tensor(mod.weight.data, cfg)))
    abs_model = copy.deepcopy(ref_model).float()
    for p in abs_model.parameters():
        p.data.abs_()

    stats = quantize_model(model, cfg)
    check("replaced == 14", stats["replaced"] == 14, str(stats))
    check("packed is smaller than fp16",
          0 < stats["packed_bytes"] < stats["fp16_bytes"],
          f"{stats['fp16_bytes'] / stats['packed_bytes']:.2f}x")
    check("lm_head untouched", isinstance(model.lm_head, nn.Linear)
          and not isinstance(model.lm_head, QuantLinear))
    check("norm untouched", isinstance(model.norm, nn.LayerNorm))
    swapped = [name for name, mod in model.named_modules()
               if isinstance(mod, QuantLinear)]
    check("all 14 swapped modules are QuantLinear", len(swapped) == 14)
    check("no nn.Linear projection left",
          not any(name.split(".")[-1].endswith("_proj")
                  and isinstance(mod, nn.Linear) and not isinstance(mod, QuantLinear)
                  for name, mod in model.named_modules()))

    x = torch.randn(1, h, dtype=torch.float16, device="cuda")
    with torch.no_grad():
        y = model(x)
        ref = ref_model(x).float()
        # Same normed input, absolute weights: an upper bound on every
        # magnitude the real forward accumulated.
        a = abs_model.body(ref_model.norm(x).float().abs())
    # 21 fp16 roundings on the path: per block, 4 projections + 2 residual-style
    # adds for q/k/v + 1 o_proj + 2 mlp projections + 1 add + 1 down_proj = 10,
    # twice, plus the lm_head. Each contributes up to the per-matmul criterion,
    # so scale it by that depth. This test is structural (did the surgery work?);
    # tests A, B and D are where the tight numeric bound lives.
    check("forward matches the fake-quant model", *criterion(y, ref, a, depth=21))


def test_graph_capture():
    print("\nD. QuantLinear forward is CUDA-graph capturable at M=1")
    torch.manual_seed(0)
    k, n = 3072, 1024
    lin = nn.Linear(k, n, bias=True).to("cuda", torch.float16)
    ql = QuantLinear.from_linear(lin, QuantConfig(4, 128, False))

    x = torch.randn(1, k, dtype=torch.float16, device="cuda")
    x2 = torch.randn(1, k, dtype=torch.float16, device="cuda")
    with torch.no_grad():
        eager1 = ql(x).clone()
        eager2 = ql(x2).clone()

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                ql(x)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_y = ql(x)

        graph.replay()
        torch.cuda.synchronize()
        check("replay equals eager", torch.equal(static_y, eager1))

        x.copy_(x2)
        graph.replay()
        torch.cuda.synchronize()
        check("replay after x.copy_ equals eager on the new input",
              torch.equal(static_y, eager2))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is required")
        sys.exit(1)

    test_gemv_reshape()
    test_quant_linear()
    test_quantize_model()
    test_graph_capture()

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)
