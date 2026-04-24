#!/usr/bin/env bash
# bench_synthetic.sh — paper-ready synthetic benchmarks: pipeline vs cuTENSOR.
#
# Three tiers with shape-specific sweeps:
#
#   Large (CUTLASS):  block-size sweep (2fA+2fB+2c tensor, section_size 8→64)
#   Medium (volume):  shape-ratio sweep at fixed MN budgets (16,64,256,1024),
#                     1fA+1fB+2c with K=4 per dim (K_total=16)
#   Small  (warp):    N sweep at M=2, three K configs (4, 16=2×4, 64=3×4)
#
# Medium and small tiers produce multiple data points per x-axis value
# for aggregation with min-max range bands in the plot.
#
# Usage:
#   ./scripts/bench_synthetic.sh                       # all tiers (uniform only)
#   ./scripts/bench_synthetic.sh --group fig1           # all tiers (same)
#   ./scripts/bench_synthetic.sh --group fig1-large     # just large tier
#   ./scripts/bench_synthetic.sh --quick                # fewer steps
#   ./scripts/bench_synthetic.sh --csv                  # also generate CSV
#   ./scripts/bench_synthetic.sh --gpu 0 --rebuild      # bind GPU, rebuild
#   ./scripts/bench_synthetic.sh --with-random          # also run ±variance variant
#
# Estimated runtime (~RTX A4500, all):  ~20 min
# Estimated runtime (H100, all):        ~8 min
#
# SLURM — uncomment to submit as a batch job:
# #SBATCH --job-name=bench-synthetic
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1
# #SBATCH --time=00:45:00
# #SBATCH --output=slurm-%j.out

BENCH_NAME="synthetic"
source "$(dirname "$0")/common.sh"

# ── Defaults ─────────────────────────────────────────────────────────────

WARMUP=1
REPEAT=3
SEED=42
GROUP="all"         # all | fig1 | fig1-large | fig1-medium | fig1-small
STEPS=10
DENSITY=0.2
DTYPE="f32"         # f32 | f64
STREAMING=128 # streaming transpose buf cap in MB
SIZE_VAR=0.3        # size variance for random variant (±30%)
UNIFORM_ONLY=true   # default: uniform shapes only; --with-random to add variance
MEM_LIMIT=""        # empty = auto-scale from GPU memory (80% of total)
MAX_PAIRS=""        # empty = auto-scale from GPU memory

# Parse benchmark-specific flags from BENCH_ARGS
setup_bench "$@"

FORWARD_ARGS=()
i=0
while [[ $i -lt ${#BENCH_ARGS[@]} ]]; do
    arg="${BENCH_ARGS[$i]}"
    case "$arg" in
        --warmup)       WARMUP="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --repeat)       REPEAT="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --seed)         SEED="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --group)        GROUP="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --steps)        STEPS="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --density)      DENSITY="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --dtype)        DTYPE="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --mem-limit)    MEM_LIMIT="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --streaming)    STREAMING="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --no-streaming) STREAMING=0; i=$((i+1)) ;;
        --max-pairs)    MAX_PAIRS="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --size-var)     SIZE_VAR="${BENCH_ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --uniform-only) UNIFORM_ONLY=true; i=$((i+1)) ;;
        --with-random)  UNIFORM_ONLY=false; i=$((i+1)) ;;
        --quick)        STEPS=6; i=$((i+1)) ;;
        *)              FORWARD_ARGS+=("$arg"); i=$((i+1)) ;;
    esac
done

# ── GPU auto-scaling ───────────────────────────────────────────────────
#
# Detect GPU memory and scale config limits relative to 20 GB baseline.
# Larger GPUs get wider sweep ranges and more pairs; streaming unchanged.
#
# Two exponents (clamped ≥1.0 — never shrink below 20 GB baseline):
#   S4 = mem^(1/4)  — safe for output tensor memory (4 free dims)
#   S6 = mem^(1/6)  — safe for 6D tensor grids (pairs ∝ nsec⁶)
#
# Example: 94 GB H100 → S4≈147%, S6≈130%.  20 GB A4500 → S4=S6=100%.

GPU_MEM_MB=20480   # fallback: 20 GB
if command -v nvidia-smi &>/dev/null; then
    _m=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
         2>/dev/null | head -1 | tr -d ' ')
    [[ -n "$_m" ]] && [[ "$_m" -gt 0 ]] 2>/dev/null && GPU_MEM_MB="$_m"
fi

S4=$(awk "BEGIN{r=${GPU_MEM_MB}/20480; printf \"%d\",((r>1?r:1)^0.25)*100+.5}")
S6=$(awk "BEGIN{r=${GPU_MEM_MB}/20480; printf \"%d\",((r>1?r:1)^(1/6.))*100+.5}")

# scale4/scale6: multiply integer by scale factor, round to nearest
scale4() { echo $(( ($1 * S4 + 50) / 100 )); }
scale6() { echo $(( ($1 * S6 + 50) / 100 )); }

# Auto-scale MEM_LIMIT and MAX_PAIRS unless explicitly set via CLI
[[ -z "$MEM_LIMIT" ]] && MEM_LIMIT=$(( GPU_MEM_MB * 80 / 100 ))
[[ -z "$MAX_PAIRS" ]] && MAX_PAIRS=$(( 200000 * GPU_MEM_MB / 20480 ))

log "GPU: $((GPU_MEM_MB / 1024)) GB — S4=${S4}% S6=${S6}% mem_limit=${MEM_LIMIT}MB max_pairs=${MAX_PAIRS}"

# ── Build ────────────────────────────────────────────────────────────────

ensure_built fuzz_pipeline

# ── Environment ──────────────────────────────────────────────────────────

capture_env

# ── Common args ──────────────────────────────────────────────────────────

COMMON=(--warmup "$WARMUP" --repeat "$REPEAT" --seed "$SEED" \
        --density "$DENSITY" --mem-limit "$MEM_LIMIT" \
        --max-pairs "$MAX_PAIRS" --verbose)
if [[ "$DTYPE" == "f64" || "$DTYPE" == "double" ]]; then
    COMMON+=(--dtype f64)
fi
[[ "$STREAMING" -gt 0 ]] && COMMON+=(--streaming "$STREAMING")
COMMON+=("${FORWARD_ARGS[@]}")

# ── Helpers ──────────────────────────────────────────────────────────────

# Run a config with uniform shapes, and optionally with random (±variance).
# Usage: run_pair "description" [fuzz_pipeline args...]
run_pair() {
    local desc="$1"; shift

    # Uniform (exact section sizes)
    run_bench "${desc} [uniform]" fuzz_pipeline --scale \
        "${COMMON[@]}" --size-variance 0 "$@"

    # Random variant (±SIZE_VAR)
    if ! $UNIFORM_ONLY; then
        run_bench "${desc} [random±${SIZE_VAR}]" fuzz_pipeline --scale \
            "${COMMON[@]}" --size-variance "$SIZE_VAR" "$@"
    fi
}

# ═════════════════════════════════════════════════════════════════════════
# Figure 1: Block-size scaling  (--sweep section_size)
#
# Fixed grid size per tier, sweep section size.  Shows per-kernel efficiency
# across block sizes.  X-axis = effective M×N (sec² for tensor, sec for mat).
# ═════════════════════════════════════════════════════════════════════════

run_fig1_large() {
    log "===== Fig 1: Large (CUTLASS) — block-size scaling ====="

    # Tensor 6D: 2fA+2fB+2c — M=N=K=sec², pairs ∝ nsec⁶
    # nsec=3, density=0.35 → ~100 pairs; sec 8→64 → M 64→4096
    # Base @20GB: nsec=3, sec 8→64
    run_pair "fig1: L-tensor 2fA+2fB+2c" --sweep section_size \
        --steps "$STEPS" \
        --num-free-a 2 --num-free-b 2 --num-contracted 2 \
        --min-section-size 8 --max-section-size "$(scale4 64)" \
        --max-sections 3 --density 0.35
}

run_fig1_medium() {
    log "===== Fig 1: Medium (volume) — shape-ratio sweep ====="

    # 1fA + 1fB + 2c, K=4 per contracted dim (K_total = 4*4 = 16).
    # At each MN budget, vary (M,N) ratios.
    # --vol-max-tiles 256 forces medium tier tile limit regardless of layout.
    # Constraints: M >= 4 (else warp tier), ceil(M/2)*ceil(N/2) <= 256,
    #              M*K_total + N*K_total <= 8192 (vol_threshold f32).
    # pairs ≈ nsec^4 * 0.04 → nsec=33 → ~47K pairs per config.
    local K=4          # per-dim section size; K_total = K^2 = 16
    local nsec=33

    local mn_list="16 64 256 1024"
    for mn in $mn_list; do
        local pairs
        case $mn in
            16)   pairs="4,4" ;;
            64)   pairs="4,16 8,8 16,4" ;;
            256)  pairs="4,64 8,32 16,16" ;;
            1024) pairs="8,128 16,64 32,32" ;;
        esac
        for p in $pairs; do
            local M=${p%,*}
            local N=${p#*,}
            run_bench "fig1: M-shape MN=${mn} M=${M} N=${N}" \
                fuzz_pipeline --scale \
                "${COMMON[@]}" --size-variance 0 \
                --sweep section_size --steps 1 \
                --num-free-a 1 --num-free-b 1 --num-contracted 2 \
                --free-a-size "$M" --free-b-size "$N" \
                --contracted-size "$K" \
                --max-sections "$nsec" \
                --vol-max-tiles 256
        done
    done
}

run_fig1_small() {
    log "===== Fig 1: Small (warp) — K-config sweep ====="

    # M=2 fixed, sweep N.  Three K configs at each N:
    #   K=4:   1 contracted dim, size 4, no shuffle (StdK, contiguous)
    #   K=16:  2 contracted dims, size 4 each (IntK, K=4×4=16)
    #   K=64:  3 contracted dims, size 4 each (IntK, K=4³=64)
    #
    # nsec is tuned per K config to equalize pair count at ~100K:
    #   pairs ≈ nsec^(num_contracted+2) × density²
    #   K=4  (nk=1): nsec=136 → 136³×0.04 ≈ 101K
    #   K=16 (nk=2): nsec=40  → 40⁴×0.04  ≈ 102K
    #   K=64 (nk=3): nsec=19  → 19⁵×0.04  ≈  99K

    local n_list="4 8 16 32 64 128 256 512"
    for N in $n_list; do
        # K=4, contiguous (1 contracted dim, no shuffle)
        run_bench "fig1: S-warp N=${N} K=4" \
            fuzz_pipeline --scale \
            "${COMMON[@]}" --size-variance 0 \
            --sweep section_size --steps 1 --tier small \
            --num-free-a 1 --num-free-b 1 --num-contracted 1 \
            --free-a-size 2 --free-b-size "$N" --contracted-size 4 \
            --no-shuffle --max-sections 136

        # K=16, multi-dim (2 contracted dims of 4, IntK)
        run_bench "fig1: S-warp N=${N} K=16" \
            fuzz_pipeline --scale \
            "${COMMON[@]}" --size-variance 0 \
            --sweep section_size --steps 1 --tier small \
            --num-free-a 1 --num-free-b 1 --num-contracted 2 \
            --free-a-size 2 --free-b-size "$N" --contracted-size 4 \
            --max-sections 40

        # K=64, higher-order (3 contracted dims of 4, IntK)
        run_bench "fig1: S-warp N=${N} K=64" \
            fuzz_pipeline --scale \
            "${COMMON[@]}" --size-variance 0 \
            --sweep section_size --steps 1 --tier small \
            --num-free-a 1 --num-free-b 1 --num-contracted 3 \
            --free-a-size 2 --free-b-size "$N" --contracted-size 4 \
            --max-sections 19
    done
}

# ── Dispatch ─────────────────────────────────────────────────────────────

run_fig1() {
    run_fig1_large
    run_fig1_medium
    run_fig1_small
}

case "$GROUP" in
    all)         run_fig1 ;;
    fig1)        run_fig1 ;;
    fig1-large)  run_fig1_large ;;
    fig1-medium) run_fig1_medium ;;
    fig1-small)  run_fig1_small ;;
    *)
        die "Unknown group: $GROUP (use: all, fig1, fig1-large, fig1-medium, fig1-small)"
        ;;
esac

# ── Done ─────────────────────────────────────────────────────────────────

finish_bench
