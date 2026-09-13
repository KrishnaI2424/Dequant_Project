# Hardware / Software Reference

This is the pinned reference sheet for the dequant-benchmarking project. Every
benchmark number reported elsewhere (achieved bandwidth, launch overhead, %-of-peak,
etc.) is normalized against the values in this file, per OUTLINE.md §4 and §9. If any
value here changes (driver update, different card, etc.), re-run the queries below and
update this file before trusting new results against old ones.

All values below were captured directly from `nvidia-smi`, `nvidia-smi -q`, and the
project's Python environment on 2026-09-12. Commands used are shown inline so the
numbers can be reproduced.

## 1. GPUs

| | GPU 0 (measurement device) | GPU 1 (compile target only) |
|---|---|---|
| Name | NVIDIA GeForce RTX 5060 Ti | NVIDIA T600 |
| Role | **The only device benchmarks are measured on** | Compile target only — never used for measurement |
| Architecture | Blackwell | Turing |
| Compute capability | 12.0 (sm_120) | 7.5 (sm_75) |
| SM count | 36 | 10 |
| VRAM | 16311 MiB (~17.10 GB decimal, see §5) | 4096 MiB (~4.29 GB decimal) |
| Max memory clock | 14001 MHz | 5001 MHz |
| Max SM clock | 3090 MHz | 2100 MHz |
| Driver model | WDDM | WDDM |
| Bus ID | 00000000:17:00.0 | 00000000:65:00.0 |

GPU 1 is excluded from measurement because it has 4 GB of VRAM and is sm_75 —
too small and too old to hold the primary model. It exists in this project only
as a second `-gencode` target so kernels compile for both cards; no benchmark
numbers from GPU 1 should ever be reported as results.

Queries used:
```
nvidia-smi --query-gpu=index,name,memory.total,clocks.max.mem,clocks.max.sm,driver_version,compute_cap --format=csv
nvidia-smi -i 0 -q -d CLOCK
```

## 2. Theoretical peak memory bandwidth (GPU 0 only)

Formula: `memory_clock_MHz × 2 (DDR) × bus_width_bits / 8 = bytes/s`, reported in
**decimal GB (1e9 bytes)**, matching this project's decimal-MB/GB convention
throughout (i.e. not MiB/GiB).

- Memory clock (max, from `nvidia-smi`): **14001 MHz**
- DDR multiplier: **×2**
- Bus width: **128-bit**. This is a manufacturer-published spec for the RTX 5060
  Ti — `nvidia-smi` does not expose memory bus width directly, so this number is
  not independently re-derivable from the tool output above and should be
  treated as an external input, not a measurement.

```
14001 MHz × 2               = 28002 million transfers/sec
28002e6 × 128 bits           = 3,584,256e6 bits/sec
3,584,256e6 bits/sec / 8     = 448,032e6 bytes/sec
                             = 448.032 GB/s (decimal, 1e9 B/s)
```

**Theoretical peak memory bandwidth: 448.032 GB/s.**

Note: the GPU uses GDDR7 signaling (PAM3), where the relationship between the
clock `nvidia-smi` reports and the actual line rate is more complex than classic
GDDR6 QDR. This calculation follows the simple clock×2×bus/8 convention this
project has standardized on (per OUTLINE.md §4); it is a nominal peak for
normalization purposes, not a vendor-certified line-rate figure. Achieved
bandwidth should always be reported as a percentage of this 448.032 GB/s number,
per OUTLINE.md §9.

## 3. Pinned software versions

| Component | Version | Notes |
|---|---|---|
| Driver | 596.36 | from `nvidia-smi` |
| CUDA (driver-reported) | 13.2 | from `nvidia-smi` |
| Python | 3.14.5 | `myenv/Scripts/python.exe --version` |
| torch | 2.12.0+cu132 | |
| transformers | 5.17.0 | |
| accelerate | 1.15.0 | |
| safetensors | 0.8.0 | |
| datasets | 5.0.1 | |
| numpy | 2.5.3 | pulled in by torch/transformers; recorded for completeness |
| torchvision | 0.27.0+cu132 | pulled in alongside torch; not directly used but pinned for env reproducibility |

Nsight tools (invoke via these exact paths — a second, older Nsight Systems
install, 2025.6.3, is also present on this machine; do not use it for this
project):

| Tool | Version | Invocation path |
|---|---|---|
| Nsight Compute | 2026.2.1.0 (build 38283040) | `C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.1/ncu.bat` |
| Nsight Systems | 2026.1.3.425 | `C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe` |

Verification commands:
```
./myenv/Scripts/python.exe -c "import torch;p=torch.cuda.get_device_properties(0);print(torch.__version__, torch.version.cuda, p.name, p.major, p.minor, p.multi_processor_count, p.total_memory)"
./myenv/Scripts/python.exe -m pip list
"C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.1/ncu.bat" --version
"C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe" --version
```

## 4. Measurement caveats

These are real limitations of this machine, not hypothetical ones. Both must be
stated alongside any benchmark result per OUTLINE.md §9.

### Clock locking is unavailable

`nvidia-smi -i 0 -lgc 3090,3090` was tested directly and fails:

```
The current user does not have permission to change clocks for GPU 00000000:17:00.0.
Terminating early due to previous errors.
```

This reproduces even from an elevated prompt. Consumer GeForce cards generally
refuse application/graphics clock locking regardless of privilege level — this
is a driver-level restriction on GeForce, not a local permissions
misconfiguration to be fixed. Because clocks cannot be locked:

- GPU 0 will boost/throttle dynamically during every benchmark run based on
  power, temperature, and workload.
- **Every reported number must be a median with spread (e.g. min/median/max or
  IQR) over repeated runs — never a single-run figure.** This is the
  "lock clocks, or state that you didn't" requirement from OUTLINE.md §9.

### WDDM driver mode inflates launch overhead

Both GPUs report `WDDM` (not TCC) in `nvidia-smi`. Consumer Blackwell (GPU 0)
cannot run TCC at all, so this is not fixable by switching driver mode.

WDDM batches command submission at the OS/driver level rather than submitting
directly to the GPU, which **inflates measured host-side launch overhead**
relative to the same code run on Linux or under TCC. This directly affects the
"launch overhead as % of decode step" Phase 0 deliverable (OUTLINE.md §9),
which in turn motivates Phase 2 (fusion) and Phase 5 (CUDA Graphs) — if that
number is measured under WDDM without comment, it will overstate the benefit
fusion/graphs would provide on other driver stacks.

**Recommendation:** do not report a single blended "launch overhead" number.
Report three separate numbers instead:
1. **Kernel-time-sum** — sum of actual on-GPU kernel execution time (from Nsight Compute/Systems).
2. **Inter-kernel gap time** — idle time between kernel end and next kernel start on the GPU timeline.
3. **Host-side time** — CPU-side submission/dispatch overhead.

This separation lets a reader discount the WDDM-specific inflation (mostly
visible in #2 and #3) when comparing against literature numbers gathered on
Linux/TCC systems.

## 5. torch vs. nvidia-smi memory reporting

`torch.cuda.get_device_properties().total_memory` and `nvidia-smi`'s memory
column do not report the same units, and this has been confirmed to cause a
visible discrepancy:

- GPU 0: `nvidia-smi` reports **16311 MiB**. `torch` reports
  `total_memory = 17102864384` bytes, i.e. **17.103 GB decimal** — but converted
  to MiB (÷1024²) that's ~16312 MiB, matching `nvidia-smi` to within rounding.
- GPU 1: `nvidia-smi` reports **4096 MiB**. `torch` reports
  `total_memory = 4294639616` bytes = **4.295 GB decimal** ≈ 4096 MiB.

In short: both tools describe the same physical memory; `nvidia-smi` displays
it in binary MiB while `torch.total_memory` is a raw byte count that looks
different if you naively divide by 1e9 (decimal) instead of 1024² (binary).
When quoting VRAM figures in this project, state which unit convention (decimal
GB vs. MiB) is being used, consistent with the decimal convention adopted for
bandwidth in §2 above.
