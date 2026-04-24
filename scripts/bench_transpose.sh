#!/usr/bin/env bash
# bench_transpose.sh — cuTENSOR vs hypertile transpose benchmark.
#
# Runs bench_transpose_cutensor with standard configurations:
#   1. Single large tensor sweep (volume x rank x regular/irregular)
#   2. Batched sweep (block_vol x block_count x rank)
#   3. Mixed-shape sweep (--mixed, opt-in)
#   4. EXP-1b(a): per-shape bandwidth with real DMRG shapes (default)
#   5. EXP-1b(b): grouped transpose with real permutations (default)
#
# Usage:
#   ./scripts/bench_transpose.sh                          # default: f32, single+batched+exp1b
#   ./scripts/bench_transpose.sh --mixed                  # include mixed-shape benchmark
#   ./scripts/bench_transpose.sh --dtype f64              # FP64
#   ./scripts/bench_transpose.sh --warmup 3 --repeat 5    # override timing params
#   ./scripts/bench_transpose.sh --csv                    # also generate CSV
#   ./scripts/bench_transpose.sh --gpu 0 --rebuild        # bind GPU, force rebuild
#   ./scripts/bench_transpose.sh --batch-sweep            # batch-sweep mode (no exp1b)
#
# SLURM — uncomment to submit as a batch job:
# #SBATCH --job-name=bench-transpose
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1
# #SBATCH --time=00:30:00
# #SBATCH --output=slurm-%j.out

BENCH_NAME="transpose"
source "$(dirname "$0")/common.sh"

# ── Defaults ─────────────────────────────────────────────────────────────

WARMUP=3
REPEAT=5
DTYPE="f32"
SEED=42
MIXED=false
BATCH_SWEEP=false
GROUPED=false
EXP1B_A=false
EXP1B_B=false
TOTAL_VOL=""

# Parse benchmark-specific flags from BENCH_ARGS
setup_bench "$@"

FORWARD_ARGS=()
i=0
while [[ $i -lt ${#BENCH_ARGS[@]} ]]; do
    arg="${BENCH_ARGS[$i]}"
    case "$arg" in
        --warmup)      WARMUP="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --repeat)      REPEAT="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --dtype)       DTYPE="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --seed)        SEED="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --mixed)       MIXED=true; i=$((i+1)) ;;
        --batch-sweep) BATCH_SWEEP=true; i=$((i+1)) ;;
        --grouped)     GROUPED=true; i=$((i+1)) ;;
        --exp1b-a)     EXP1B_A=true; i=$((i+1)) ;;
        --exp1b-b)     EXP1B_B=true; i=$((i+1)) ;;
        --total-vol)   TOTAL_VOL="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        *)             FORWARD_ARGS+=("$arg"); i=$((i+1)) ;;
    esac
done

# ── Build ────────────────────────────────────────────────────────────────

ensure_built bench_transpose_cutensor

# ── Environment ──────────────────────────────────────────────────────────

capture_env

# ── Run ──────────────────────────────────────────────────────────────────

COMMON_ARGS=(--warmup "$WARMUP" --repeat "$REPEAT" --seed "$SEED" "${FORWARD_ARGS[@]}")

if [[ "$DTYPE" == "f64" || "$DTYPE" == "double" ]]; then
    COMMON_ARGS+=(--dtype f64)
fi

if $MIXED; then
    COMMON_ARGS+=(--mixed)
fi

if $BATCH_SWEEP; then
    COMMON_ARGS+=(--batch-sweep)
fi

if $GROUPED; then
    COMMON_ARGS+=(--grouped "../generated_tensors/cutensor_converted")
fi

# EXP-1b runs by default (unless --batch-sweep is the primary mode)
if ! $BATCH_SWEEP; then
    EXP1B_A=true
    EXP1B_B=true
fi

if $EXP1B_A; then
    COMMON_ARGS+=(--exp1b-a)
fi

if $EXP1B_B; then
    COMMON_ARGS+=(--exp1b-b "../generated_tensors/cutensor_converted")
fi

if [[ -n "$TOTAL_VOL" ]]; then
    COMMON_ARGS+=(--total-vol "$TOTAL_VOL")
fi

run_bench "transpose (${DTYPE}${MIXED:+, mixed})" \
    bench_transpose_cutensor "${COMMON_ARGS[@]}"

# ── Done ─────────────────────────────────────────────────────────────────

finish_bench
