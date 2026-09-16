"""Ahead-of-time build for the Phase 1 dequantization kernels.

    python setup.py build_ext --inplace

Drops the compiled .pyd next to this file. Re-run after editing kernels/.
MSVC is located explicitly rather than assumed on PATH, matching CUDA/make.ps1.
"""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

KERNELS = Path(__file__).parent / "kernels"

# sm_120 = Blackwell (RTX 5060 Ti), the only measurement device. sm_75 (T600)
# is kept as a compile target so the build stays honest on both cards.
nvcc_args = [
    "-gencode", "arch=compute_75,code=sm_75",
    "-gencode", "arch=compute_120,code=sm_120",
    "-O3",
]
if os.name == "nt":
    # CUDA 13's CCCL headers reject MSVC's legacy preprocessor.
    nvcc_args += ["-Xcompiler", "/Zc:preprocessor"]

setup(
    name="dequant_cuda",
    ext_modules=[
        CUDAExtension(
            name="dequant_cuda",
            sources=[
                str(KERNELS / "smoke.cu"),
                str(KERNELS / "dequant_gemv.cu"),
                str(KERNELS / "bindings.cpp"),
            ],
            extra_compile_args={"cxx": [], "nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
