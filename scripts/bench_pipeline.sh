#!/usr/bin/env bash
# bench_pipeline.sh — pipeline implementation benchmark (CUTLASS vs cuTENSOR).
#
# Runs bench_pipeline_implementations with standard configurations and
# captures output for later analysis.
#
# Usage:
#   ./scripts/bench_pipeline.sh                          # default: f32 full suite
#   ./scripts/bench_pipeline.sh --dtype f64              # FP64 subset
#   ./scripts/bench_pipeline.sh --filter nf_E3           # filter by case name
#   ./scripts/bench_pipeline.sh --warmup 3 --repeat 5    # override timing params
#   ./scripts/bench_pipeline.sh --csv                    # also generate CSV
#   ./scripts/bench_pipeline.sh --gpu 0 --rebuild        # bind GPU, force rebuild
#
# All flags not consumed by common.sh are forwarded to the binary.
#
# SLURM — uncomment to submit as a batch job:
# #SBATCH --job-name=bench-pipeline
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1
# #SBATCH --time=01:00:00
# #SBATCH --output=slurm-%j.out

BENCH_NAME="pipeline"
source "$(dirname "$0")/common.sh"

# ── Defaults ─────────────────────────────────────────────────────────────

WARMUP=3
REPEAT=5
DTYPE="f32"
FILTER=""
STREAMING=128  # streaming transpose buf cap in MB
EXTRA_FLAGS=(--no-verify)

# Parse benchmark-specific flags from BENCH_ARGS (remainder after common.sh)
setup_bench "$@"

FORWARD_ARGS=()
i=0
while [[ $i -lt ${#BENCH_ARGS[@]} ]]; do
    arg="${BENCH_ARGS[$i]}"
    case "$arg" in
        --warmup)  WARMUP="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --repeat)  REPEAT="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --dtype)   DTYPE="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --filter|-f) FILTER="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --verify)  EXTRA_FLAGS=(); i=$((i+1)) ;;   # override --no-verify
        --streaming)    STREAMING="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --no-streaming) STREAMING=0; i=$((i+1)) ;;
        *)         FORWARD_ARGS+=("$arg"); i=$((i+1)) ;;
    esac
done

# ── Build ────────────────────────────────────────────────────────────────

ensure_built bench_pipeline_implementations

# ── Environment ──────────────────────────────────────────────────────────

capture_env

# ── Run ──────────────────────────────────────────────────────────────────

COMMON_ARGS=(--warmup "$WARMUP" --repeat "$REPEAT" "${EXTRA_FLAGS[@]}")
[[ "$STREAMING" -gt 0 ]] && COMMON_ARGS+=(--streaming "$STREAMING")
COMMON_ARGS+=("${FORWARD_ARGS[@]}")

if [[ "$DTYPE" == "f64" || "$DTYPE" == "double" ]]; then
    COMMON_ARGS+=(--dtype f64)
fi

if [[ -n "$FILTER" ]]; then
    run_bench "pipeline (${DTYPE}, filter=${FILTER})" \
        bench_pipeline_implementations "${COMMON_ARGS[@]}" --filter "$FILTER"
else
    run_bench "pipeline (${DTYPE}, full suite)" \
        bench_pipeline_implementations "${COMMON_ARGS[@]}"
fi

# ── Done ─────────────────────────────────────────────────────────────────

finish_bench
