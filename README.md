# TIDES

GPU implementation for block-sparse tensor contraction. Dispatches each block pair to
one of three execution regimes, warp-level (registers), shared-memory tiled,
or grouped GEMM and streams transpose work through a bounded
workspace that overlaps with contraction.

The grouped-GEMM (large) tier ranks two backends per shape queue, CUTLASS
grouped GEMM and cuBLAS `<t>gemmGroupedBatched`, using a per-GPU efficiency
model; see *Large-tier backend selection* below. The warp tier routes
scalar-output, large-K block pairs (Lanczos inner products) to a dedicated
CTA-per-pair reduction kernel.

## Prerequisites

**Hardware**
- NVIDIA H100 NVL (SM 9.0, 94 GB HBM3) on x86_64 with ≥128 GB system memory.
  H100 80 GB (SXM/PCIe) also runs the full suite; smaller Hopper/Ampere GPUs
  may skip cases that exceed HBM.

**Software**
- CUDA Toolkit 12.6, driver 560.35.05. **12.5 is the minimum**: the
  large-tier cuBLAS backend calls `cublas{S,D}gemmGroupedBatched`, added in
  12.5. Also verified with CUDA 13.0.
- GCC 11.5 (or a newer C++17-capable compiler)
- CMake ≥ 3.23
- Python 3 with `matplotlib`, `numpy` (for plotting)
- [cuTENSOR 2.6.0](https://developer.nvidia.com/cutensor-downloads) 
  baseline and correctness oracle
- [CUTLASS v4.3.0](https://github.com/NVIDIA/cutlass) at commit `a4eb0e05`,
  with the patch from `patches/` applied (see below)

## Setup

### 1. CUTLASS with the PermuteD patch

CUTLASS's `PredicatedTileIterator` miscomputes addresses in the `PermuteD`
epilogue. Apply the patch shipped in this repo:

```bash
git clone https://github.com/NVIDIA/cutlass.git ../cutlass
git -C ../cutlass checkout a4eb0e05
git -C ../cutlass apply $(pwd)/patches/cutlass-predicated-tile-iterator-permuteD-load.patch
```

CMake auto-discovers CUTLASS at `../cutlass`; pass `-DCUTLASS_ROOT=/path`
to override.

### 2. cuTENSOR

Install cuTENSOR 2.6.0 and symlink (or copy) the unpacked archive into this
repo as `libcutensor/`, or pass `-DCUTENSOR_ROOT=/path/to/cutensor` to CMake.

### 3. Build

The full configure command, with every path given explicitly. Run it from the
top of this repository, on a **fresh** `build/` directory:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBSPTC_ARCH=sm90 \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCUTLASS_ROOT=$(pwd)/../cutlass \
  -DCUTENSOR_ROOT=$(pwd)/libcutensor
cmake --build build -j
```

For Ampere use `-DBSPTC_ARCH=sm80 -DCMAKE_CUDA_ARCHITECTURES=80`.
`BSPTC_ARCH` accepts any `smXX` nvcc does, so a minor version such as
`-DBSPTC_ARCH=sm86` (RTX A4500) or an architecture-specific `-DBSPTC_ARCH=sm90a`
works too, and sets `CMAKE_CUDA_ARCHITECTURES` to match. Drop
`-DCUTLASS_ROOT` / `-DCUTENSOR_ROOT` if the dependencies are already at the
default locations from steps 1 and 2. The build produces the benchmark and
test binaries under `build/`.

`-DCMAKE_CUDA_ARCHITECTURES` is redundant with `-DBSPTC_ARCH` on a fresh build
directory, and is included above because it is the one setting that is
authoritative in every situation. `BSPTC_ARCH` is a convenience that maps to
it, and CMake ignores that mapping if `CMAKE_CUDA_ARCHITECTURES` is already
cached from an earlier configure, or if `CUDAARCHS` is set in the environment
(some HPC module files set it). Both cases are now reported at configure time.

**Verify before building**: the configure output must contain

```
-- CUDA architectures: 90 (BSPTC_ARCH=sm90)
```

If it names a different architecture, stop and fix it — the build either fails
with `no instance of overloaded function "atomicAdd" matches the argument
list` (architectures below SM 6.0 have no `atomicAdd` for `double`), or, worse,
succeeds for the wrong target and silently PTX-JITs at run time, which
invalidates every timing number it produces. After building, confirm the real
thing:

```bash
cuobjdump --list-elf build/bench_pipeline_implementations | grep -o 'sm_[0-9]*' | sort -u
```

### 4. Large-tier backend selection

Each large-tier shape queue picks between CUTLASS and cuBLAS grouped GEMM by
scoring both against a calibrated per-GPU efficiency table (`include/general_sparse/tile_efficiency.hpp`, calibrated for SM 8.6 and SM 9.0; other
architectures fall back to CUTLASS unconditionally). No action is needed for
the reported runs; the following override the choice for ablation:

```bash
BSPTC_LARGE_BACKEND=cutlass|cublas|auto   # force a backend (default: auto)
BSPTC_DUMP_KERNEL_CHOICE=1                # print per-queue scores and winner
BSPTC_DUMP_SHAPE_PLAN=<minK>              # dump the per-shape tier table
```

Recalibrate the efficiency table on a new GPU with
`./build/bench_cutlass_tiles --cublas` and `--cublas-groups`.

## Experiments

The three benchmark scripts correspond to the three evaluations reported in
the paper. All scripts write timestamped output under `results/`.

### E1 — DMRG block-sparse contractions (TIDES vs cuTENSOR)

```bash
./scripts/bench_pipeline.sh --dtype f64 --csv
./scripts/bench_pipeline.sh --dtype f32 --csv
```

Runs the 60 DMRG cases bundled in `generated_tensors/cutensor_converted/` on
both TIDES and cuTENSOR's block-sparse API, median of 5 timed runs after 3
warmup iterations. Add `--filter <substring>` to restrict to a subset.

### E2 — Per-tier synthetic sweeps

```bash
./scripts/bench_synthetic.sh --dtype f64 --csv
./scripts/bench_synthetic.sh --dtype f32 --csv
```

Isolates each dispatch tier (warp, medium, large/grouped-GEMM) with synthetic
shapes.

### E3 — Grouped transpose vs cuTENSOR

```bash
./scripts/bench_transpose.sh --dtype f64 --csv           # batched (one shape)
./scripts/bench_transpose.sh --dtype f64 --mixed --csv   # grouped (heterogeneous)
```

## Analysis

Each bench script emits a raw `output.txt` plus (with `--csv`) a parsed
`results.csv` under `results/<benchmark>/<YYYY-MM-DD>_<commit>_<host>/`.
The CSV column `tides_speedup` is `cutensor / tides`, so values above 1 mean
TIDES is faster.

Regenerate the paper figures:

```bash
python3 scripts/plot_benchmarks.py           # figures for E1, E2, E3
python3 scripts/plot_synthetic_blocksize.py  # synthetic block-size sweep
```

Plots auto-discover the latest run under `results/`; override with
`--pipe-f32 DIR --pipe-f64 DIR …` or the `TIDES_RESULTS_BASE` env var.

### Setup-cost measurement

`bench_pipeline_implementations` reports setup for both implementations at the
same boundary, on a `[prepare]` line:

```
[prepare] tides_status=ok tides_host_prepare_ms=... tides_device_zero_ms=...
          tides_prepare_ms=... cutensor_status=ok cutensor_host_prepare_ms=...
          cutensor_device_zero_ms=... cutensor_prepare_ms=...
```

The measured interval starts after input analysis and device allocation and
ends when the implementation is launch-ready: `host_prepare_ms` is planning
and metadata staging, `device_zero_ms` is the accumulator zero-fill, and
`prepare_ms` is their sum. Both sides allocate outside the timed region and
zero their accumulators inside it, so the two numbers are directly
comparable. The `[wall]` line is a diagnostic only, since cuTENSOR is handed
precomputed output coordinates; `[setup_detail]` is a TIDES-only breakdown.

## Tests

```bash
./build/test_pipeline_verification              # 117 correctness cases
./build/fuzz_pipeline --fuzz --count 1000 --dtype f32   # random-shape fuzzer
./build/fuzz_pipeline --fuzz --count 1000 --dtype f64
```

Reproduce a fuzzer failure with `--seed <N> --count 1 --verbose`.

## Layout

```
src/            pipeline, dispatch, CUTLASS and cuBLAS grouped-GEMM backends
include/        headers
tests/          correctness/verification + benchmark entrypoints
scripts/        benchmark wrappers (bench_pipeline, bench_synthetic,
                bench_transpose) and plotting
patches/        CUTLASS PermuteD fix
generated_tensors/cutensor_converted/
                60 bundled DMRG contraction cases: three formulations
                (canonical, block-permuted internal-only, block-permuted
                matricized) × four symmetry regimes (kysznf, sznf, nf,
                nfparity) × five problem sizes
```
