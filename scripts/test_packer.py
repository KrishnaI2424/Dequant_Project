"""Verification gate for the offline packer.

Checks properties that must hold for ANY correct quantizer, rather than
eyeballing whether the error "looks small":

  1. pack/unpack is exactly lossless, every bit width x layout.
  2. Reconstruction error never exceeds the theoretical bound (scale/2 for
     symmetric, scale for asymmetric), for every format combination.
  3. Error shrinks monotonically as bit width grows.
  4. Error shrinks monotonically as group size shrinks.
  5. Asymmetric beats symmetric on skewed data -- the case it exists for.
  6. Layout is purely a storage detail: sequential and interleaved must
     dequantize to bit-identical values.

    python scripts/test_packer.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dequant.packer import (
    QuantConfig,
    dequantize,
    pack,
    quantize,
    quantize_tensor,
    unpack,
)

BITS = (2, 4, 8)
LAYOUTS = ("sequential", "interleaved")
GROUPS = (32, 64, 128, 256, None)

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def test_pack_roundtrip():
    print("\n1. pack/unpack is lossless")
    torch.manual_seed(0)
    for bits in BITS:
        for layout in LAYOUTS:
            cfg = QuantConfig(bits=bits, layout=layout)
            q = torch.randint(0, cfg.qmax + 1, (8, 512), dtype=torch.int32)
            back = unpack(pack(q, cfg), cfg, 512)
            check(f"{bits}-bit {layout}", torch.equal(q, back))


def test_error_bound():
    print("\n2. error within theoretical bound")
    torch.manual_seed(0)
    w = torch.randn(64, 512, dtype=torch.float16)
    for bits in BITS:
        for sym in (True, False):
            for gs in GROUPS:
                cfg = QuantConfig(bits=bits, group_size=gs, symmetric=sym)
                pw = quantize_tensor(w, cfg)
                err = (dequantize(pw).float() - w.float()).abs()

                # Round-to-nearest cannot err by more than half a step. The
                # asymmetric path also clamps, so allow a full step there.
                gsz = cfg.effective_group_size(512)
                step = pw.scales.float().repeat_interleave(gsz, dim=1)
                bound = step * (0.5 if sym else 1.0) + 1e-3
                worst = (err / bound).max().item()
                label = f"{bits}-bit g{gs} {'sym' if sym else 'asym'}"
                check(label, worst <= 1.0, f"worst={worst:.3f}x bound")


def test_monotonic_bits():
    print("\n3. more bits -> less error")
    torch.manual_seed(0)
    w = torch.randn(64, 512, dtype=torch.float16)
    errs = []
    for bits in BITS:
        pw = quantize_tensor(w, QuantConfig(bits=bits, group_size=128))
        errs.append((dequantize(pw).float() - w.float()).abs().mean().item())
    print(f"     mean abs err: " + ", ".join(f"{b}-bit {e:.5f}"
                                             for b, e in zip(BITS, errs)))
    check("monotonic in bits", all(a > b for a, b in zip(errs, errs[1:])))


def test_monotonic_group():
    print("\n4. smaller groups -> less error")
    torch.manual_seed(0)
    w = torch.randn(64, 512, dtype=torch.float16)
    sizes = (32, 64, 128, 256, None)
    errs = []
    for gs in sizes:
        pw = quantize_tensor(w, QuantConfig(bits=4, group_size=gs))
        errs.append((dequantize(pw).float() - w.float()).abs().mean().item())
    print("     mean abs err: " + ", ".join(f"g{s} {e:.5f}"
                                            for s, e in zip(sizes, errs)))
    check("monotonic in group size", all(a < b for a, b in zip(errs, errs[1:])))


def test_asymmetric_on_skew():
    print("\n5. asymmetric beats symmetric on skewed data")
    torch.manual_seed(0)
    # All-positive, offset from zero: exactly the distribution symmetric
    # quantization handles badly, since it must waste half its codes.
    w = (torch.rand(64, 512) * 0.5 + 2.0).to(torch.float16)
    e_sym = (dequantize(quantize_tensor(w, QuantConfig(4, 128, True))).float()
             - w.float()).abs().mean().item()
    e_asym = (dequantize(quantize_tensor(w, QuantConfig(4, 128, False))).float()
              - w.float()).abs().mean().item()
    print(f"     symmetric {e_sym:.5f}  asymmetric {e_asym:.5f}"
          f"  ({e_sym / e_asym:.1f}x better)")
    check("asymmetric wins on skew", e_asym < e_sym)


def test_layout_equivalence():
    print("\n6. layout does not change the values")
    torch.manual_seed(0)
    w = torch.randn(64, 512, dtype=torch.float16)
    for bits in BITS:
        a = dequantize(quantize_tensor(w, QuantConfig(bits, 128, True, "sequential")))
        b = dequantize(quantize_tensor(w, QuantConfig(bits, 128, True, "interleaved")))
        check(f"{bits}-bit layouts agree", torch.equal(a, b))


def test_real_weight_shapes():
    print("\n7. shapes the 3B model actually uses")
    # Element-wise relative error for 4-bit symmetric on Gaussian weights is
    # ~13% and that is CORRECT, not a bug: with group max ~2.9 sigma over 128
    # samples, scale ~= 2.9/7 sigma, mean rounding error = scale/4 ~= 0.10
    # sigma, against mean |w| = 0.80 sigma. So rather than assert an arbitrary
    # threshold, check the measured error matches that prediction -- a much
    # sharper test, and one that would catch a genuinely broken quantizer.
    torch.manual_seed(0)
    # Llama-3.2-3B: hidden 3072, intermediate 8192, q_dim 3072, kv_dim 1024.
    shapes = [(3072, 3072), (1024, 3072), (8192, 3072), (3072, 8192)]
    for n, k in shapes:
        w = torch.randn(n, k, dtype=torch.float16) * 0.02
        pw = quantize_tensor(w, QuantConfig(4, 128, True))
        err = (dequantize(pw).float() - w.float()).abs().mean().item()
        predicted = pw.scales.float().mean().item() / 4
        ratio = err / predicted

        fp16_mb = n * k * 2 / 1e6
        packed_mb = pw.nbytes / 1e6
        rel = err / w.float().abs().mean().item()
        check(f"[{n},{k}]", 0.85 <= ratio <= 1.15,
              f"err {err:.2e} vs predicted {predicted:.2e} ({ratio:.2f}x), "
              f"rel {rel * 100:.1f}%, {fp16_mb:.1f}->{packed_mb:.1f} MB "
              f"({fp16_mb / packed_mb:.2f}x)")


if __name__ == "__main__":
    test_pack_roundtrip()
    test_error_bound()
    test_monotonic_bits()
    test_monotonic_group()
    test_asymmetric_on_skew()
    test_layout_equivalence()
    test_real_weight_shapes()

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)
