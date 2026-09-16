"""Correctness gate for the CUDA dequant-GEMV kernel.

The kernel and dequant/packer.py must agree bit for bit, so the tests are
exact where they can be and bounded by an arithmetic argument where they
cannot:

  1. The extension still imports and add_one survives the bindings move.
  2. dequantize() is torch.equal to packer.dequantize() over every
     bits x layout x symmetry x group combination. This is the test that
     catches an interleave-table mistake, so it runs before any GEMV test.
  3. gemv() matches an fp32 reference within a bound derived from fp32
     rounding, not from eyeballing.
  4. Negative control: one flipped weight bit must break test 3's criterion.
  5. The kernel is capturable into a CUDA graph and replays against the live
     input buffer.

    python scripts/test_dequant_kernel.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dequant_cuda  # noqa: E402  (the .pyd sits at the project root)

from dequant import packer  # noqa: E402
from dequant.packer import QuantConfig, quantize_tensor  # noqa: E402

BITS = (2, 4, 8)
LAYOUTS = ("sequential", "interleaved")
GROUPS = (32, 64, 128, 256, None)

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def ext_args(pw):
    """The trailing arguments both extension entry points take."""
    k = pw.shape[1]
    cfg = pw.config
    return (pw.qweight, pw.scales, pw.zeros, cfg.bits,
            cfg.effective_group_size(k), cfg.symmetric,
            cfg.layout == "interleaved")


def test_extension():
    print("\n1. extension loads")
    print(f"     {dequant_cuda.__file__}")
    out = dequant_cuda.add_one(torch.ones(4, device="cuda"))
    check("add_one", torch.equal(out, torch.full((4,), 2.0, device="cuda")))


def test_dequantize_bit_exact():
    print("\n2. dequantize is bit-exact with packer.dequantize")
    torch.manual_seed(0)
    n, k = 1024, 3072
    gauss = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
    # All-positive and offset from zero, so asymmetric zero-points are large
    # and the (q - zero) subtraction goes negative for most codes.
    skew = (torch.rand(n, k, device="cuda") * 0.5 + 2.0).to(torch.float16)

    for bits in BITS:
        for layout in LAYOUTS:
            for sym in (True, False):
                for gs in GROUPS:
                    cases = [("gauss", gauss)] if sym else [("gauss", gauss),
                                                            ("skew", skew)]
                    for tag, w in cases:
                        cfg = QuantConfig(bits, gs, sym, layout)
                        pw = quantize_tensor(w, cfg)
                        got = dequant_cuda.dequantize(*ext_args(pw), k)
                        want = packer.dequantize(pw)
                        label = (f"{bits}-bit {layout[:3]} "
                                 f"{'sym' if sym else 'asym'} g{gs} {tag}")
                        if torch.equal(got, want):
                            check(label, True)
                        else:
                            bad = (got != want).sum().item()
                            check(label, False, f"{bad}/{got.numel()} words differ")


def exact_weight(pw):
    """(code - offset) * scale in fp32, WITHOUT packer.dequantize's fp16 store.

    The kernel never materializes the weight: it holds (code - offset) * scale
    in a register and multiplies straight into the fp32 accumulator. Comparing
    it against packer.dequantize()'s fp16 output would charge it for that
    output's own 2^-11 rounding, which for K=3072 is ~2^-17 of the absolute
    product sum -- 10-16x the fp32 accumulation budget below, so the reference
    would dominate the tolerance and the test would measure nothing. Test 2
    already gates the packed format itself bit-exactly; this reference isolates
    the dot product.
    """
    cfg = pw.config
    n, k = pw.shape
    gs = cfg.effective_group_size(k)
    q = packer.unpack(pw.qweight, cfg, k).reshape(n, k // gs, gs).float()
    off = cfg.bias if cfg.symmetric else pw.zeros.float().unsqueeze(-1)
    return ((q - off) * pw.scales.float().unsqueeze(-1)).reshape(n, k)


def gemv_criterion(y, ref, absw, absx):
    """|y - ref| <= 2^-11 |ref| + 2^-20 (|W| @ |x|^T), elementwise.

    The second term is the fp32 accumulation budget: the kernel sums K products
    in a lane-serial + warp-tree order, so its rounding error scales with the
    sum of absolute products, not with the (cancelled) result.
    """
    a = absw.float() @ absx.float().t()          # [N, M]
    tol = 2.0 ** -11 * ref.abs() + 2.0 ** -20 * a.t()
    err = (y.float() - ref).abs()
    return err, tol


def run_gemv_case(n, k, cfg, m, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
    pw = quantize_tensor(w, cfg)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    we = exact_weight(pw)
    y = dequant_cuda.gemv(x, *ext_args(pw))
    ref = F.linear(x.float(), we)
    err, tol = gemv_criterion(y, ref, we.abs(), x.abs())
    return y, ref, err, tol


GEMV_FORMATS = [
    ("INT4 g128 sym  seq", QuantConfig(4, 128, True, "sequential")),
    ("INT4 g128 asym seq", QuantConfig(4, 128, False, "sequential")),
    ("INT4 g128 sym  int", QuantConfig(4, 128, True, "interleaved")),
    ("INT4 g128 asym int", QuantConfig(4, 128, False, "interleaved")),
    ("INT8 g128 sym  seq", QuantConfig(8, 128, True, "sequential")),
    ("INT8 g128 sym  int", QuantConfig(8, 128, True, "interleaved")),
    ("INT4 per-ch  sym  seq", QuantConfig(4, None, True, "sequential")),
]
GEMV_SHAPES = [(3072, 3072), (1024, 3072), (8192, 3072), (3072, 8192)]


def test_gemv(ms):
    print(f"\n3. gemv vs fp32 reference (M in {ms})")
    torch.backends.cuda.matmul.allow_tf32 = False
    for m in ms:
        for n, k in GEMV_SHAPES:
            for label, cfg in GEMV_FORMATS:
                y, ref, err, tol = run_gemv_case(n, k, cfg, m)
                ok = bool((err <= tol).all())
                exact = (y == ref.half()).float().mean().item()
                rel = (err / ref.abs().clamp_min(1e-30)).max().item()
                head = (err / tol).max().item()
                check(f"M={m} [{n},{k}] {label}",
                      ok, f"max|e|={err.max().item():.3e} maxrel={rel:.2e} "
                          f"{head:.2f}x tol exact={exact * 100:.1f}%")

        # INT2 g64 at K=3072: 3072/64 = 48 chunks over 32 lanes, so the
        # per-lane loop guard (not a clean multiple of the warp) is exercised.
        cfg = QuantConfig(2, 64, True, "sequential")
        y, ref, err, tol = run_gemv_case(3072, 3072, cfg, m)
        ok = bool((err <= tol).all())
        exact = (y == ref.half()).float().mean().item()
        check(f"M={m} [3072,3072] INT2 g64 sym seq", ok,
              f"max|e|={err.max().item():.3e} {(err / tol).max().item():.2f}x tol "
              f"exact={exact * 100:.1f}%")

    print("\n3b. excluded and out-of-range formats raise")
    cfg = QuantConfig(2, 32, True, "sequential")
    pw = quantize_tensor(torch.randn(64, 256, device="cuda", dtype=torch.float16), cfg)
    x = torch.randn(1, 256, device="cuda", dtype=torch.float16)
    try:
        dequant_cuda.gemv(x, *ext_args(pw))
        check("INT2 g32 rejected", False, "no exception")
    except RuntimeError as exc:
        check("INT2 g32 rejected", True, str(exc).splitlines()[0][:90])

    cfg = QuantConfig(4, 128, True, "sequential")
    pw = quantize_tensor(torch.randn(64, 256, device="cuda", dtype=torch.float16), cfg)
    x = torch.randn(9, 256, device="cuda", dtype=torch.float16)
    try:
        dequant_cuda.gemv(x, *ext_args(pw))
        check("M=9 rejected", False, "no exception")
    except RuntimeError as exc:
        check("M=9 rejected", True, str(exc).splitlines()[0][:90])


def test_negative_control():
    print("\n4. negative control: one flipped weight bit must fail the criterion")
    torch.manual_seed(0)
    n, k = 3072, 3072
    cfg = QuantConfig(4, 128, True, "sequential")
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
    pw = quantize_tensor(w, cfg)
    x = torch.zeros(1, k, device="cuda", dtype=torch.float16)
    x.normal_()
    x[0, 0] = 1.0

    we = exact_weight(pw)
    ref = F.linear(x.float(), we)

    bad = pw.qweight.clone()
    bad[0, 0] ^= 1
    y = dequant_cuda.gemv(x, bad, pw.scales, pw.zeros, cfg.bits,
                          cfg.effective_group_size(k), cfg.symmetric, False)
    err, tol = gemv_criterion(y, ref, we.abs(), x.abs())
    check("flipped bit is detected at [0,0]", bool(err[0, 0] > tol[0, 0]),
          f"err={err[0, 0].item():.3e} tol={tol[0, 0].item():.3e} "
          f"({(err[0, 0] / tol[0, 0]).item():.1f}x)")
    check("only element [0,0] is affected", int((err > tol).sum()) == 1,
          f"{int((err > tol).sum())} elements over tolerance")


def test_graph_capture():
    print("\n5. CUDA graph capture")
    torch.manual_seed(0)
    n, k = 3072, 3072
    cfg = QuantConfig(4, 128, False, "sequential")
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
    pw = quantize_tensor(w, cfg)
    args = ext_args(pw)
    x = torch.randn(1, k, device="cuda", dtype=torch.float16)
    x2 = torch.randn(1, k, device="cuda", dtype=torch.float16)

    eager1 = dequant_cuda.gemv(x, *args).clone()
    eager2 = dequant_cuda.gemv(x2, *args).clone()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            dequant_cuda.gemv(x, *args)
    torch.cuda.current_stream().wait_stream(side)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = dequant_cuda.gemv(x, *args)

    g.replay()
    torch.cuda.synchronize()
    check("replay matches eager", torch.equal(y, eager1))

    x.copy_(x2)
    g.replay()
    torch.cuda.synchronize()
    check("replay reads the live input buffer", torch.equal(y, eager2))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is required")
        sys.exit(1)

    only = sys.argv[1] if len(sys.argv) > 1 else "all"

    if only in ("all", "1"):
        test_extension()
    if only in ("all", "2"):
        test_dequantize_bit_exact()
    if only in ("all", "3"):
        test_gemv((1, 2, 3, 8))
    elif only == "3-m1":
        test_gemv((1,))
    if only in ("all", "4"):
        test_negative_control()
    if only in ("all", "5"):
        test_graph_capture()

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)
