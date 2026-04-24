#!/usr/bin/env bash
# common.sh — shared framework for benchmark scripts.
# Source this at the top of every bench_*.sh script.
#
# Provides:
#   setup_bench       Parse common flags, locate build dir, create output dir
#   ensure_built      Build target if binary is missing or --rebuild requested
#   capture_env       Write environment snapshot to env.txt
#   log               Timestamped log message
#   run_bench         Execute a benchmark binary, tee to output file
#   finish_bench      Print summary, optionally run parser
#
# Common flags (parsed by setup_bench):
#   --rebuild          Force rebuild before running
#   --dry-run          Print commands without executing
#   --output-dir DIR   Override output directory
#   --gpu ID           Set CUDA_VISIBLE_DEVICES
#   --csv              Run parse_results.py after benchmarking

set -euo pipefail

# ── Globals (set by setup_bench) ─────────────────────────────────────────

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

BENCH_NAME="${BENCH_NAME:-}"  # set by caller before sourcing common.sh
OUTPUT_DIR=""        # resolved by setup_bench
OUTPUT_FILE=""       # path to raw output
ENV_FILE=""          # path to env snapshot
ARGS_FILE=""         # path to argument record

OPT_REBUILD=false
OPT_DRY_RUN=false
OPT_CSV=false
OPT_GPU=""
OPT_OUTPUT_DIR=""
BENCH_EXIT=0          # tracks worst exit code across all run_bench calls

# ── Helpers ──────────────────────────────────────────────────────────────

log() {
    printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*"
}

die() {
    printf "[ERROR] %s\n" "$*" >&2
    exit 1
}

# ── setup_bench ──────────────────────────────────────────────────────────
# Call after setting BENCH_NAME.  Parses common flags from "$@" and sets
# up the output directory.  Returns remaining (benchmark-specific) args
# in the BENCH_ARGS array.

setup_bench() {
    [[ -n "$BENCH_NAME" ]] || die "BENCH_NAME must be set before calling setup_bench"

    BENCH_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --rebuild)     OPT_REBUILD=true; shift ;;
            --dry-run)     OPT_DRY_RUN=true; shift ;;
            --csv)         OPT_CSV=true; shift ;;
            --gpu)         OPT_GPU="$2"; shift 2 ;;
            --output-dir)  OPT_OUTPUT_DIR="$2"; shift 2 ;;
            *)             BENCH_ARGS+=("$1"); shift ;;
        esac
    done

    # GPU binding
    if [[ -n "$OPT_GPU" ]]; then
        export CUDA_VISIBLE_DEVICES="$OPT_GPU"
        log "CUDA_VISIBLE_DEVICES=$OPT_GPU"
    fi

    # Resolve output directory
    local commit
    commit="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")"
    local datestamp
    datestamp="$(date '+%Y-%m-%d')"
    local host
    host="$(hostname -s)"

    local timestamp
    timestamp="$(date '+%H%M%S')"
    local run_tag="${datestamp}_${commit}_${host}_${timestamp}"

    if [[ -n "$OPT_OUTPUT_DIR" ]]; then
        OUTPUT_DIR="$OPT_OUTPUT_DIR"
    else
        OUTPUT_DIR="$REPO_ROOT/results/${BENCH_NAME}/${run_tag}"
    fi

    mkdir -p "$OUTPUT_DIR"
    OUTPUT_FILE="$OUTPUT_DIR/output.txt"
    ENV_FILE="$OUTPUT_DIR/env.txt"
    ARGS_FILE="$OUTPUT_DIR/args.txt"

    log "Benchmark: $BENCH_NAME"
    log "Output:    $OUTPUT_DIR"
}

# ── ensure_built ─────────────────────────────────────────────────────────
# Usage: ensure_built target_name [target_name ...]

ensure_built() {
    local targets=("$@")

    if $OPT_REBUILD; then
        log "Rebuilding: ${targets[*]}"
        make -C "$BUILD_DIR" "${targets[@]}" -j"$(nproc)" 2>&1 | tail -5
    else
        for t in "${targets[@]}"; do
            if [[ ! -x "$BUILD_DIR/$t" ]]; then
                log "Binary not found, building: $t"
                make -C "$BUILD_DIR" "$t" -j"$(nproc)" 2>&1 | tail -5
            fi
        done
    fi

    for t in "${targets[@]}"; do
        [[ -x "$BUILD_DIR/$t" ]] || die "Build failed: $BUILD_DIR/$t not found"
    done
}

# ── capture_env ──────────────────────────────────────────────────────────

capture_env() {
    {
        echo "date:      $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "hostname:  $(hostname)"
        echo "user:      $(whoami)"
        echo "git_branch: $(git -C "$REPO_ROOT" branch --show-current 2>/dev/null || echo 'detached')"
        echo "git_commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo 'unknown')"
        echo "git_dirty:  $(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | head -1 | grep -q . && echo yes || echo no)"

        if command -v nvidia-smi &>/dev/null; then
            echo ""
            echo "gpu_name:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
            echo "gpu_driver:  $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
            echo "gpu_memory:  $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
        fi

        if command -v nvcc &>/dev/null; then
            echo "cuda_version: $(nvcc --version 2>/dev/null | grep 'release' | sed 's/.*release //' | sed 's/,.*//')"
        fi

        echo ""
        echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
        echo "build_type: $(grep CMAKE_BUILD_TYPE "$BUILD_DIR/CMakeCache.txt" 2>/dev/null | cut -d= -f2 || echo unknown)"
        echo "cuda_arch:  $(grep CMAKE_CUDA_ARCHITECTURES "$BUILD_DIR/CMakeCache.txt" 2>/dev/null | cut -d= -f2 || echo unknown)"
    } > "$ENV_FILE"

    log "Environment captured: $ENV_FILE"
}

# ── run_bench ────────────────────────────────────────────────────────────
# Usage: run_bench description binary [args...]
# Runs the binary from BUILD_DIR, tees output to OUTPUT_FILE.
# Appends to OUTPUT_FILE so multiple runs accumulate.

run_bench() {
    local desc="$1"; shift
    local binary="$1"; shift

    local cmd=("./$binary" "$@")
    local cmd_str="${cmd[*]}"

    # Record arguments
    echo "$desc: $cmd_str" >> "$ARGS_FILE"

    log "Running: $desc"
    log "  (in $BUILD_DIR) $cmd_str"

    if $OPT_DRY_RUN; then
        log "  (dry-run, skipped)"
        return 0
    fi

    echo "" >> "$OUTPUT_FILE"
    echo "# === $desc ===" >> "$OUTPUT_FILE"
    echo "# cmd: $cmd_str" >> "$OUTPUT_FILE"
    echo "# cwd: $BUILD_DIR" >> "$OUTPUT_FILE"
    echo "# started: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    # Run from BUILD_DIR (binaries expect data paths relative to build/)
    (cd "$BUILD_DIR" && "${cmd[@]}") 2>&1 | tee -a "$OUTPUT_FILE"
    local rc=${PIPESTATUS[0]}

    echo "" >> "$OUTPUT_FILE"
    echo "# exit_code: $rc" >> "$OUTPUT_FILE"
    echo "# finished: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"

    if [[ $rc -ne 0 ]]; then
        log "WARNING: $binary exited with code $rc (continuing)"
        BENCH_EXIT=$rc
    fi

    return 0
}

# ── finish_bench ─────────────────────────────────────────────────────────

finish_bench() {
    log "Done. Output: $OUTPUT_DIR"

    if $OPT_CSV; then
        local parser="$SCRIPTS_DIR/parse_results.py"
        if [[ -f "$parser" ]]; then
            local csv_file="$OUTPUT_DIR/results.csv"
            log "Generating CSV: $csv_file"
            python3 "$parser" "$OUTPUT_FILE" --csv -o "$csv_file"
        else
            log "WARNING: parse_results.py not found, skipping CSV"
        fi
    fi

    log "Files:"
    ls -lh "$OUTPUT_DIR"/ | tail -n+2 | while read -r line; do log "  $line"; done

    if [[ $BENCH_EXIT -ne 0 ]]; then
        log "WARNING: one or more benchmarks failed (exit $BENCH_EXIT)"
    fi
    return $BENCH_EXIT
}
