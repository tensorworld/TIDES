# Benchmark Scripts

Bash scripts for reproducible GPU benchmarking, designed for HPC environments.

## Layout

```
scripts/
  common.sh              Shared framework (sourced by all bench scripts)
  bench_pipeline.sh      Pipeline: CUTLASS vs cuTENSOR block-sparse contraction
  bench_transpose.sh     Transpose: cuTENSOR permutation vs hypertile kernel
  bench_synthetic.sh     Synthetic tier benchmarks: warp + volume vs cuTENSOR
  parse_results.py       Parse raw output → CSV or aligned table
  README.md              This file
```

## Quick start

```bash
# From repo root — run the pipeline benchmark with defaults
./scripts/bench_pipeline.sh

# Transpose benchmark with mixed-shape mode
./scripts/bench_transpose.sh --mixed

# Generate CSV alongside raw output
./scripts/bench_pipeline.sh --csv

# Parse an existing output file to CSV
python3 scripts/parse_results.py results/pipeline/.../output.txt -o results.csv

# Human-readable table
python3 scripts/parse_results.py results/transpose/.../output.txt --format table
```

## Output convention

Each run creates a timestamped directory under `results/`:

```
results/<benchmark>/<YYYY-MM-DD>_<commit>_<hostname>/
  env.txt        Environment snapshot (GPU, CUDA, git, host)
  args.txt       Exact commands executed
  output.txt     Raw benchmark stdout
  results.csv    Parsed CSV (if --csv was passed)
```

The directory name encodes the date, git commit, and hostname so runs are
self-documenting and diffable across machines or code revisions.

## Common flags

These are handled by `common.sh` and available in every `bench_*.sh` script:

| Flag             | Description                              |
|------------------|------------------------------------------|
| `--rebuild`      | Force `make` before running              |
| `--dry-run`      | Print commands without executing          |
| `--csv`          | Run `parse_results.py` after benchmarking |
| `--gpu ID`       | Set `CUDA_VISIBLE_DEVICES`               |
| `--output-dir D` | Override the output directory             |

All other flags are forwarded to the benchmark binary.

## Writing a new benchmark script

1. Copy an existing `bench_*.sh` as a template.
2. Set `BENCH_NAME` to a short identifier (used in the output path).
3. Call `setup_bench "$@"` to parse common flags.
4. Call `ensure_built <target>` to build the binary.
5. Call `capture_env` to snapshot the environment.
6. Call `run_bench <description> <binary> [args...]` for each run.
7. Call `finish_bench` to print summary and optionally generate CSV.

Benchmark-specific flags are parsed from `BENCH_ARGS` (the remainder after
`common.sh` consumes its own flags).

## SLURM

Each script includes commented-out `#SBATCH` headers. Uncomment and adjust
for your cluster, then submit with `sbatch scripts/bench_pipeline.sh`.

## parse_results.py

Autodetects the benchmark format from section headers in the output file.
No external dependencies (stdlib only).

```bash
# CSV to stdout
python3 scripts/parse_results.py output.txt

# CSV to file
python3 scripts/parse_results.py output.txt -o results.csv

# Aligned text table
python3 scripts/parse_results.py output.txt --format table
```

Pipeline CSV columns: `case_name, einsum, a_blocks, b_blocks, layout,
cutlass_ms, cutensor_ms, speedup, transpose_ms, contract_ms, accumulate_ms,
large_problems, medium_descs, small_descs, total_flops_gflop, peak_memory_mb`

Transpose CSV columns vary by section (`single`, `batched`, `mixed`);
each row has a `section` column to distinguish them.
