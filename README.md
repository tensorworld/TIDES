# TIDES

GPU implementation for block-sparse tensor contraction. Dispatches each block pair to
one of three execution regimes, warp-level (registers), shared-memory tiled,
or CUTLASS grouped GEMM and streams transpose work through a bounded
workspace that overlaps with contraction.

## Prerequisites

**Hardware**
- NVIDIA H100 NVL (SM 9.0, 94 GB HBM3) on x86_64 with ≥128 GB system memory.
  H100 80 GB (SXM/PCIe) also runs the full suite; smaller Hopper/Ampere GPUs
  may skip cases that exceed HBM.

**Software**
- CUDA Toolkit 12.6, driver 560.35.05
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

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBSPTC_ARCH=sm90
cmake --build build -j
```

Use `-DBSPTC_ARCH=sm80` for Ampere. The build produces the benchmark and
test binaries under `build/`.

## Experiments

The three benchmark scripts correspond to the three evaluations reported in
the paper. All scripts write timestamped output under `results/`.

### E1 — DMRG block-sparse contractions (TIDES vs cuTENSOR)

```bash
./scripts/bench_pipeline.sh --dtype f64 --csv
./scripts/bench_pipeline.sh --dtype f32 --csv
```

Runs the 40 DMRG cases bundled in `generated_tensors/cutensor_converted/` on
both TIDES and cuTENSOR's block-sparse API, median of 5 timed runs after 3
warmup iterations.

### E2 — Per-tier synthetic sweeps

```bash
./scripts/bench_synthetic.sh --dtype f64 --csv
./scripts/bench_synthetic.sh --dtype f32 --csv
```

Isolates each dispatch tier (warp, medium, large/CUTLASS) with synthetic
shapes.

### E3 — Grouped transpose vs cuTENSOR

```bash
./scripts/bench_transpose.sh --dtype f64 --csv           # batched (one shape)
./scripts/bench_transpose.sh --dtype f64 --mixed --csv   # grouped (heterogeneous)
```

## Analysis

Each bench script emits a raw `output.txt` plus (with `--csv`) a parsed
`results.csv` under `results/<benchmark>/<YYYY-MM-DD>_<commit>_<host>/`.

Regenerate the paper figures:

```bash
python3 scripts/plot_benchmarks.py           # figures for E1, E2, E3
python3 scripts/plot_synthetic_blocksize.py  # synthetic block-size sweep
```

Plots auto-discover the latest run under `results/`; override with
`--pipe-f32 DIR --pipe-f64 DIR …` or the `TIDES_RESULTS_BASE` env var.

## Tests

```bash
./build/test_pipeline_verification              # correctness on bundled cases
./build/fuzz_pipeline --count 1000 --dtype f32  # random-shape fuzzer
./build/fuzz_pipeline --count 1000 --dtype f64
```

## Layout

```
src/            pipeline, dispatch, CUTLASS grouped-GEMM instantiations
include/        headers
tests/          correctness/verification + benchmark entrypoints
scripts/        benchmark wrappers (bench_pipeline, bench_synthetic,
                bench_transpose) and plotting
patches/        CUTLASS PermuteD fix
generated_tensors/cutensor_converted/
                40 bundled DMRG contraction cases across four symmetry
                regimes (kysznf, sznf, nf, nfparity) × two layouts
```
