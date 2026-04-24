#!/usr/bin/env python3
"""Plot benchmark results.

Auto-discovers the latest result directories under ./results/ (override with
$TIDES_RESULTS_BASE), or accepts explicit paths via CLI flags.

Generates:
  1. Synthetic Fig1: kernel speedup vs M/N-equiv (exec only, no setup)
  2. Synthetic Fig2: kernel speedup vs pair count (exec only)
  3. Setup time scaling with pair count (dedicated figure)
  4. Pipeline: bar chart capped at 5x, with geomean lines
  5. Transpose: hypertile vs cuTENSOR bandwidth comparison
  6. Batch sweep: transpose speedup vs batch size (fixed + mixed, f32 + f64)
  7. Tier distribution: stacked bar chart of FLOPs fraction per tier (EXP-4)
  8. Case study: grouped speedup bars by symmetry x size x layout (EXP-3)

Usage:
    python3 scripts/plot_benchmarks.py                           # auto-discover latest
    python3 scripts/plot_benchmarks.py --synth-dir DIR           # explicit synthetic dir
    python3 scripts/plot_benchmarks.py --pipe-f32 DIR --pipe-f64 DIR --trans-f32 DIR --trans-f64 DIR
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from math import exp, log

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm

# ── Defaults ───────────────────────────────────────────────────────────

RESULTS_BASE = os.environ.get("TIDES_RESULTS_BASE", "results")

# Populated by main() from args or auto-discovery
COMMIT = ""
GPU_LABEL = ""

# ── Auto-discovery ─────────────────────────────────────────────────────

def _list_result_dirs(category, commit_prefix=None):
    """List result dirs for a category, newest first. Optionally filter by commit prefix."""
    base = os.path.join(RESULTS_BASE, category)
    if not os.path.isdir(base):
        return []
    dirs = sorted(glob.glob(os.path.join(base, "*")), reverse=True)
    results = []
    for d in dirs:
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "results.csv")):
            continue
        if commit_prefix and f"_{commit_prefix}" not in os.path.basename(d):
            continue
        results.append(d)
    return results


def _latest_dir(category, commit_prefix=None):
    """Find the most recent result directory for a category (by name sort)."""
    dirs = _list_result_dirs(category, commit_prefix)
    return dirs[0] if dirs else None


def _find_dirs_by_dtype(category, commit_prefix=None, require_flag=None,
                        exclude_flag=None):
    """Find f32 and f64 result dirs for a category. Returns (f32_dir, f64_dir).

    require_flag: only match dirs whose args.txt contains this string.
    exclude_flag: skip dirs whose args.txt contains this string.
    """
    dirs = _list_result_dirs(category, commit_prefix)
    f32_dir = f64_dir = None
    for d in dirs:
        args_path = os.path.join(d, "args.txt")
        if not os.path.exists(args_path):
            continue
        with open(args_path) as f:
            args_line = f.readline()
        if require_flag and require_flag not in args_line:
            continue
        if exclude_flag and exclude_flag in args_line:
            continue
        if f32_dir is None and "f32" in args_line:
            f32_dir = d
        elif f64_dir is None and "f64" in args_line:
            f64_dir = d
        if f32_dir and f64_dir:
            break
    return f32_dir, f64_dir


def _read_env(result_dir):
    """Extract commit hash and GPU label from env.txt."""
    env_path = os.path.join(result_dir, "env.txt")
    commit = gpu = ""
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("git_commit:"):
                    commit = line.split(":", 1)[1].strip()[:7]
                elif line.startswith("gpu_name:"):
                    gpu = line.split(":", 1)[1].strip()
                elif line.startswith("gpu_memory:"):
                    mem = line.split(":", 1)[1].strip()
                    # "95830 MiB" -> "95 GB"
                    m = re.match(r"(\d+)", mem)
                    if m:
                        gpu += f" {int(m.group(1)) // 1024} GB"
    return commit, gpu


# ── Helpers ────────────────────────────────────────────────────────────

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def parse_synth_csv(path):
    rows = read_csv(path)
    for r in rows:
        for k in ("param", "exec_ms", "ct_kern_ms", "setup_ms", "ct_setup_ms",
                   "exec_speedup", "total_speedup"):
            if k in r:
                try:    r[k] = float(r[k])
                except (ValueError, TypeError): r[k] = 0.0
        for k in ("pairs", "large", "medium", "small", "M", "N", "K"):
            if k in r:
                try:    r[k] = int(r[k])
                except (ValueError, TypeError): r[k] = 0
    return rows

def group_by_config(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["config"]].append(r)
    return groups

def geomean(values):
    """Geometric mean of positive values."""
    pos = [v for v in values if v > 0]
    if not pos:
        return 0
    return exp(sum(log(v) for v in pos) / len(pos))

# ── M/N-equiv from section_size ────────────────────────────────────────

def get_mn_equiv(r):
    """Get max(M, N) from CSV columns."""
    return max(r.get("M", 0), r.get("N", 0))

# ── Config tables ──────────────────────────────────────────────────────

FIG1_CONFIGS = [
    ("fig1: L-tensor 2fA+2fB+2c [uniform]",     "L-tensor (6D) uni", "#1f77b4", "-"),
    ("fig1: L-tensor 2fA+2fB+2c [random±0.3]",  "L-tensor (6D) rnd", "#1f77b4", "--"),
    ("fig1: L-matrix 1fA+1fB+1c [uniform]",      "L-matrix (2D) uni", "#ff7f0e", "-"),
    ("fig1: L-matrix 1fA+1fB+1c [random±0.3]",   "L-matrix (2D) rnd", "#ff7f0e", "--"),
    ("fig1: M-tensor 2fA+2fB+2c [uniform]",      "M-tensor (6D) uni", "#2ca02c", "-"),
    ("fig1: M-tensor 2fA+2fB+2c [random±0.3]",   "M-tensor (6D) rnd", "#2ca02c", "--"),
    ("fig1: M-matrix 1fA+1fB+1c [uniform]",      "M-matrix (2D) uni", "#d62728", "-"),
    ("fig1: M-matrix 1fA+1fB+1c [random±0.3]",   "M-matrix (2D) rnd", "#d62728", "--"),
    ("fig1: S-tensor 1fA+2fB+1c [uniform]",      "S-tensor (4D) uni", "#9467bd", "-"),
    ("fig1: S-tensor 1fA+2fB+1c [random±0.3]",   "S-tensor (4D) rnd", "#9467bd", "--"),
    ("fig1: S-matrix 1fA+1fB+1c [uniform]",      "S-matrix (2D) uni", "#8c564b", "-"),
    ("fig1: S-matrix 1fA+1fB+1c [random±0.3]",   "S-matrix (2D) rnd", "#8c564b", "--"),
]

FIG2_CONFIGS = [
    ("fig2: L-tensor 2fA+2fB+2c sec=16 [uniform]",      "L-tensor sec=16 uni", "#1f77b4", "-"),
    ("fig2: L-tensor 2fA+2fB+2c sec=16 [random±0.3]",   "L-tensor sec=16 rnd", "#1f77b4", "--"),
    ("fig2: L-matrix 1fA+1fB+1c sec=512 [uniform]",     "L-matrix sec=512 uni", "#ff7f0e", "-"),
    ("fig2: L-matrix 1fA+1fB+1c sec=512 [random±0.3]",  "L-matrix sec=512 rnd", "#ff7f0e", "--"),
    ("fig2: M-tensor 2fA+2fB+2c sec=4 [uniform]",       "M sec=4 uni",  "#2ca02c", "-"),
    ("fig2: M-tensor 2fA+2fB+2c sec=4 [random±0.3]",    "M sec=4 rnd",  "#2ca02c", "--"),
    ("fig2: S-tensor 1fA+2fB+1c sec=8 [uniform]",       "S sec=8 uni",  "#9467bd", "-"),
    ("fig2: S-tensor 1fA+2fB+1c sec=8 [random±0.3]",    "S sec=8 rnd",  "#9467bd", "--"),
]

# ── Plotting ───────────────────────────────────────────────────────────

def good_pts(groups, config_name):
    return [r for r in groups.get(config_name, [])
            if r["status"] == "PASS" and r["exec_ms"] > 0]


def plot_fig1_by_tier(groups, out_dir):
    """Fig1: exec-only kernel speedup vs M/N-equiv, split by tier, with geomean."""
    tiers = [
        ("Large (CUTLASS)", [c for c in FIG1_CONFIGS if c[1].startswith("L-")]),
        ("Medium (volume)", [c for c in FIG1_CONFIGS if c[1].startswith("M-")]),
        ("Small (warp)",    [c for c in FIG1_CONFIGS if c[1].startswith("S-")]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, (tier_name, configs) in zip(axes, tiers):
        all_speedups = []
        for config_name, label, color, ls in configs:
            pts = good_pts(groups, config_name)
            if not pts:
                continue
            xs = [get_mn_equiv(r) for r in pts]
            ys = [r["exec_speedup"] for r in pts]
            all_speedups.extend(ys)
            short = label.replace("L-","").replace("M-","").replace("S-","")
            ax.plot(xs, ys, color=color, linestyle=ls, marker="o", markersize=4,
                    label=short, linewidth=1.5)

        ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("M or N equiv", fontsize=10)
        ax.set_title(f"Fig1 {tier_name}", fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    axes[0].set_ylabel("Kernel speedup (ct_kern / exec)", fontsize=10)
    fig.suptitle(f"Fig 1: Block-Size Scaling (f32, kernel only) — {GPU_LABEL} ({COMMIT})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_block_size_by_tier.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_fig2_combined(groups, out_dir):
    """Fig2: exec-only kernel speedup vs pair count, with geomean."""
    fig, ax = plt.subplots(figsize=(10, 6))

    all_speedups = []
    for config_name, label, color, ls in FIG2_CONFIGS:
        pts = [r for r in good_pts(groups, config_name) if r["pairs"] > 0]
        if not pts:
            continue
        xs = [r["pairs"] for r in pts]
        ys = [r["exec_speedup"] for r in pts]
        all_speedups.extend(ys)
        ax.plot(xs, ys, color=color, linestyle=ls, marker="o", markersize=4,
                label=label, linewidth=1.5)

    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Number of block pairs", fontsize=11)
    ax.set_ylabel("Kernel speedup (ct_kern / exec)", fontsize=11)
    ax.set_title(f"Fig 2: Pair-Count Scaling (f32, kernel only) — {GPU_LABEL} ({COMMIT})",
                 fontsize=13)
    ax.set_xscale("log")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_pair_count.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_setup_scaling(groups, out_dir):
    """Dedicated figure: setup time vs pair count for all fig2 configs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for config_name, label, color, ls in FIG2_CONFIGS:
        pts = [r for r in good_pts(groups, config_name) if r["pairs"] > 0]
        if not pts:
            continue
        xs = [r["pairs"] for r in pts]
        ys = [r["setup_ms"] for r in pts]
        ax.plot(xs, ys, color=color, linestyle=ls, marker="o", markersize=4,
                label=label, linewidth=1.5)
    ax.set_xlabel("Number of block pairs", fontsize=10)
    ax.set_ylabel("Pipeline setup time (ms)", fontsize=10)
    ax.set_title("Pipeline Setup", fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for config_name, label, color, ls in FIG2_CONFIGS:
        pts = [r for r in good_pts(groups, config_name) if r["pairs"] > 0]
        if not pts:
            continue
        xs = [r["pairs"] for r in pts]
        ys = [r["ct_setup_ms"] for r in pts]
        ax.plot(xs, ys, color=color, linestyle=ls, marker="o", markersize=4,
                label=label, linewidth=1.5)
    ax.set_xlabel("Number of block pairs", fontsize=10)
    ax.set_ylabel("cuTENSOR setup time (ms)", fontsize=10)
    ax.set_title("cuTENSOR Setup", fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Setup Time vs Pair Count (f32) — {GPU_LABEL} ({COMMIT})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "setup_time_scaling.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_fig1_setup_vs_mn(groups, out_dir):
    """Setup time vs M/N-equiv for fig1 uniform configs."""
    fig1_uniform = [c for c in FIG1_CONFIGS if "uniform" in c[0]]

    fig, ax = plt.subplots(figsize=(10, 6))
    for config_name, label, color, ls in fig1_uniform:
        pts = good_pts(groups, config_name)
        if not pts:
            continue
        xs = [get_mn_equiv(r) for r in pts]
        ys = [r["setup_ms"] for r in pts]
        ax.plot(xs, ys, color=color, linestyle=ls, marker="o", markersize=4,
                label=label, linewidth=1.5)

    ax.set_xlabel("M or N equiv", fontsize=11)
    ax.set_ylabel("Pipeline setup time (ms)", fontsize=11)
    ax.set_title(f"Pipeline Setup Time vs Block Size (f32, uniform) — {GPU_LABEL} ({COMMIT})",
                 fontsize=12)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_setup_time.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Pipeline bar chart ─────────────────────────────────────────────────

PIPE_CAP = 5.0  # cap bar height; show true value as text


def _load_cherry_pick(arg):
    """Parse cherry-pick arg: comma-separated substrings or @filename. Returns list or None."""
    if not arg:
        return None
    if arg.startswith('@'):
        with open(arg[1:]) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [s.strip() for s in arg.split(',') if s.strip()]


def _match_cherry(name, picks):
    """True if name matches any substring in picks."""
    return any(p in name for p in picks)


def plot_pipeline(f32_csv, f64_csv, out_dir, cherry_pick=None):
    """Bar chart: pipeline vs cuTENSOR, capped at PIPE_CAP, geomean line.

    speedup in CSV = cutlass_ms / cutensor_ms (lower = pipeline faster).
    Plot cuTENSOR_ms / pipeline_ms so >1 = pipeline wins.
    """
    f32 = read_csv(f32_csv)
    f64 = read_csv(f64_csv)

    def parse_case(rows):
        results = []
        for r in rows:
            name = r["case_name"]
            if cherry_pick and not _match_cherry(name, cherry_pick):
                continue
            cutlass = float(r["cutlass_ms"]) if r["cutlass_ms"] else 0
            cutensor = float(r["cutensor_ms"]) if r["cutensor_ms"] else 0
            if cutlass == 0 and cutensor == 0:
                continue
            su = cutensor / cutlass if cutlass > 0 else 0
            results.append({
                "name": name,
                "speedup": su,
                "cutlass_ms": cutlass,
                "cutensor_ms": cutensor,
            })
        return results

    f32_cases = parse_case(f32)
    f64_cases = parse_case(f64)

    def categorize(name):
        if "canonical" in name:       return "can"
        if "block_permuted_matrices" in name: return "mat"
        if "block_permuted_internal" in name: return "int"
        return "other"

    def short_name(name):
        # e.g. "block_permuted_internal_only_symm_nf_E2" -> "int/nf_E2"
        cat = categorize(name)
        parts = name.split("_symm_")
        suffix = parts[1] if len(parts) == 2 else name[-15:]
        return f"{cat}/{suffix}"

    for dtype_label, cases in [("f32", f32_cases), ("f64", f64_cases)]:
        if not cases:
            continue

        cats = defaultdict(list)
        for c in cases:
            cats[categorize(c["name"])].append(c)

        ordered = []
        cat_boundaries = []
        for cat in ["can", "int", "mat"]:
            if cat in cats:
                items = sorted(cats[cat], key=lambda x: x["name"])
                cat_boundaries.append((len(ordered), cat))
                ordered.extend(items)

        if not ordered:
            continue

        fig, ax = plt.subplots(figsize=(16, 6))
        names = [short_name(c["name"]) for c in ordered]
        speedups = [c["speedup"] for c in ordered]

        # Cap bars at PIPE_CAP
        bar_heights = [min(s, PIPE_CAP) for s in speedups]
        colors = ["#2ca02c" if s >= 1 else "#d62728" for s in speedups]

        bars = ax.bar(range(len(names)), bar_heights, color=colors, alpha=0.8, width=0.7)

        # Labels: show true value on capped bars, skip for uncapped
        for bar, s, bh in zip(bars, speedups, bar_heights):
            if s > PIPE_CAP:
                ax.text(bar.get_x() + bar.get_width()/2, bh + 0.05,
                        f"{s:.1f}x", ha="center", va="bottom", fontsize=7,
                        fontweight="bold", color="#1a5e1a")

        # Geomean line
        gm = geomean(speedups)
        if gm > 0:
            ax.axhline(y=gm, color="blue", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.text(len(names) - 0.5, gm + 0.05, f"gmean={gm:.2f}x",
                    color="blue", fontsize=9, ha="right", va="bottom")

        ax.axhline(y=1.0, color="black", linestyle="-", linewidth=0.8)
        ax.set_ylim(0, PIPE_CAP + 0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("Speedup (cuTENSOR / pipeline)", fontsize=11)
        ax.set_title(f"Pipeline Benchmark ({dtype_label}) — {GPU_LABEL} ({COMMIT})",
                     fontsize=13)
        ax.grid(True, axis="y", alpha=0.3)

        # Category separators + labels
        for i, (start, cat) in enumerate(cat_boundaries):
            if start > 0:
                ax.axvline(x=start-0.5, color="gray", linestyle="--",
                           linewidth=0.5, alpha=0.5)
            end = cat_boundaries[i+1][0] if i+1 < len(cat_boundaries) else len(ordered)
            mid = (start + end) / 2
            ax.text(mid, -0.35, cat, ha="center", va="top", fontsize=8,
                    color="gray", transform=ax.get_xaxis_transform())

        fig.tight_layout()
        path = os.path.join(out_dir, f"pipeline_{dtype_label}.pdf")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


# ── Transpose plots ───────────────────────────────────────────────────

SECTION_COLORS = {
    "single":  "#1f77b4",
    "batched": "#ff7f0e",
    "mixed":   "#2ca02c",
}

def parse_transpose_csv(path):
    rows = read_csv(path)
    for r in rows:
        for k in ("cutensor_ms", "hypertile_ms", "speedup", "ht_gbps", "ct_gbps"):
            if k not in r:
                continue
            try:    r[k] = float(r[k])
            except (ValueError, TypeError): r[k] = 0.0
        for k in ("num_blocks", "avg_vol", "rank", "block_vol"):
            if k not in r:
                continue
            try:    r[k] = int(r[k])
            except (ValueError, TypeError): r[k] = 0
    return rows


def _transpose_label(r):
    """Build a compact, readable label for one transpose case."""
    section = r["section"]
    perm = r.get("perm", "")
    if section == "single":
        shape = r.get("shape", "")
        # Shorten perm: (1,2,0) -> p120
        short_perm = "p" + perm.replace(",", "")
        return f"{shape}  {short_perm}"
    elif section == "batched":
        blk = r.get("block_shape", "") or r.get("shape", "")
        n = r.get("num_blocks", 0)
        short_perm = "p" + perm.replace(",", "")
        return f"{blk} x{n}  {short_perm}"
    else:  # mixed
        n = r.get("num_blocks", 0)
        avg = r.get("avg_vol", 0)
        return f"mixed x{n}  avg={avg}"


def plot_transpose(f32_csv, f64_csv, out_dir):
    """Transpose benchmark: grouped bandwidth bar chart, one per dtype."""
    for dtype_label, csv_path in [("f32", f32_csv), ("f64", f64_csv)]:
        if not os.path.exists(csv_path):
            continue
        rows = parse_transpose_csv(csv_path)
        if not rows:
            continue

        # Group by section, keep only valid rows
        by_section = defaultdict(list)
        for r in rows:
            if r["ht_gbps"] > 0:
                by_section[r["section"]].append(r)

        # Build ordered list with section boundaries
        ordered = []
        section_bounds = []  # (start_idx, section_name, count)
        for section in ["single", "batched", "mixed"]:
            pts = by_section.get(section, [])
            if not pts:
                continue
            section_bounds.append((len(ordered), section, len(pts)))
            ordered.extend(pts)

        if not ordered:
            continue

        n = len(ordered)
        labels = [_transpose_label(r) for r in ordered]
        ht_bw = [r["ht_gbps"] for r in ordered]
        ct_bw = [r["ct_gbps"] for r in ordered]
        speedups = [r["speedup"] for r in ordered]

        # ── Figure ──
        fig_w = max(14, n * 0.30)
        fig, ax = plt.subplots(figsize=(fig_w, 6))

        x = list(range(n))
        w = 0.38
        ax.bar([i - w/2 for i in x], ct_bw, w,
               label="cuTENSOR", color="#e07050", alpha=0.85,
               edgecolor="white", linewidth=0.3)
        ax.bar([i + w/2 for i in x], ht_bw, w,
               label="Hypertile", color="#4caf50", alpha=0.85,
               edgecolor="white", linewidth=0.3)

        # Speedup annotations — only for notable cases (>=2x or <=0.5x)
        for i, su in enumerate(speedups):
            if su < 2.0 and su > 0.5:
                continue
            top = max(ht_bw[i], ct_bw[i])
            color = "#2e7d32" if su >= 1.0 else "#c62828"
            label_text = f"{su:.0f}x" if su >= 10 else f"{su:.1f}x"
            ax.text(i, top + 40, label_text, ha="center", va="bottom",
                    fontsize=6, color=color, fontweight="bold")

        # Section separators and background shading
        for j, (start, sec_name, count) in enumerate(section_bounds):
            end = start + count
            if start > 0:
                ax.axvline(x=start - 0.5, color="#555", linestyle="-",
                           linewidth=1.2)
            # Alternate light background
            if j % 2 == 1:
                ax.axvspan(start - 0.5, end - 0.5, color="#f0f0f0",
                           alpha=0.5, zorder=0)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
        ax.set_ylabel("Bandwidth (GB/s)", fontsize=11)
        ax.set_title(f"Transpose: Hypertile vs cuTENSOR ({dtype_label}) — "
                     f"{GPU_LABEL} ({COMMIT})", fontsize=13, pad=15)
        ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        ax.set_xlim(-0.6, n - 0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Section labels centered in each group, placed just inside top
        ymax = ax.get_ylim()[1]
        for start, sec_name, count in section_bounds:
            mid = (start + start + count - 1) / 2
            ax.text(mid, ymax * 0.95, sec_name.upper(), ha="center", va="top",
                    fontsize=11, fontweight="bold", color="#444",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#ccc", alpha=0.9))

        fig.tight_layout()
        path = os.path.join(out_dir, f"transpose_{dtype_label}.pdf")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_exp1b(f32_csv, f64_csv, out_dir):
    """EXP-1b: per-shape bandwidth (a) and grouped real-perm (b) on one figure."""
    for dtype_label, csv_path in [("f32", f32_csv), ("f64", f64_csv)]:
        if not os.path.exists(csv_path):
            continue
        rows = parse_transpose_csv(csv_path)
        if not rows:
            continue

        rows_a = [r for r in rows if r.get("section") == "exp1b_a" and r.get("ht_gbps", 0) > 0]
        rows_b = [r for r in rows if r.get("section") == "exp1b_b" and r.get("ht_gbps", 0) > 0]

        if not rows_a and not rows_b:
            continue

        num_panels = (1 if rows_a else 0) + (1 if rows_b else 0)
        fig, axes = plt.subplots(1, num_panels, figsize=(max(14, 7 * num_panels), 6))
        if num_panels == 1:
            axes = [axes]
        panel_idx = 0

        # ── Panel (a): per-shape ──
        if rows_a:
            ax = axes[panel_idx]; panel_idx += 1
            n = len(rows_a)
            labels = [f"{r['shape']}\n{r['perm']} WR={r['wr']}" for r in rows_a]
            ht_bw = [r["ht_gbps"] for r in rows_a]
            ct_bw = [r["ct_gbps"] for r in rows_a]
            speedups = [r["speedup"] for r in rows_a]

            x = list(range(n))
            w = 0.38
            ax.bar([i - w/2 for i in x], ct_bw, w,
                   label="cuTENSOR", color="#e07050", alpha=0.85,
                   edgecolor="white", linewidth=0.3)
            ax.bar([i + w/2 for i in x], ht_bw, w,
                   label="Hypertile", color="#4caf50", alpha=0.85,
                   edgecolor="white", linewidth=0.3)

            for i, su in enumerate(speedups):
                top = max(ht_bw[i], ct_bw[i])
                color = "#2e7d32" if su >= 1.0 else "#c62828"
                label_text = f"{su:.1f}x"
                ax.text(i, top + 20, label_text, ha="center", va="bottom",
                        fontsize=6, color=color, fontweight="bold")

            # Shape group separators
            shapes_seen = []
            for i, r in enumerate(rows_a):
                s = r["shape"]
                if not shapes_seen or shapes_seen[-1] != s:
                    if shapes_seen:
                        ax.axvline(x=i - 0.5, color="#555", linestyle="-", linewidth=0.8)
                    shapes_seen.append(s)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)
            ax.set_ylabel("Bandwidth (GB/s)", fontsize=11)
            ax.set_title(f"(a) Per-Shape Bandwidth ({dtype_label})", fontsize=12, pad=10)
            ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
            ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
            ax.set_xlim(-0.6, n - 0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # ── Panel (b): grouped real permutations ──
        if rows_b:
            ax = axes[panel_idx]; panel_idx += 1
            n = len(rows_b)
            # Short label: symmetry + size + operand
            def short_name(r):
                cn = r["case_name"]
                # Extract symmetry_size suffix
                for prefix in ["canonical_formulation_symm_"]:
                    cn = cn.replace(prefix, "")
                return f"{cn} {r['operand']}"
            labels = [short_name(r) for r in rows_b]
            ht_bw = [r["ht_gbps"] for r in rows_b]
            ct_bw = [r["ct_gbps"] for r in rows_b]
            speedups = [r["speedup"] for r in rows_b]

            x = list(range(n))
            w = 0.38
            ax.bar([i - w/2 for i in x], ct_bw, w,
                   label="cuTENSOR", color="#e07050", alpha=0.85,
                   edgecolor="white", linewidth=0.3)
            ax.bar([i + w/2 for i in x], ht_bw, w,
                   label="Hypertile", color="#4caf50", alpha=0.85,
                   edgecolor="white", linewidth=0.3)

            for i, su in enumerate(speedups):
                if su < 1.5 and su > 0.67:
                    continue
                top = max(ht_bw[i], ct_bw[i])
                color = "#2e7d32" if su >= 1.0 else "#c62828"
                label_text = f"{su:.0f}x" if su >= 10 else f"{su:.1f}x"
                ax.text(i, top + 20, label_text, ha="center", va="bottom",
                        fontsize=6, color=color, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)
            ax.set_ylabel("Bandwidth (GB/s)", fontsize=11)
            ax.set_title(f"(b) Grouped Real Permutations ({dtype_label})", fontsize=12, pad=10)
            ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
            ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
            ax.set_xlim(-0.6, n - 0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(f"EXP-1b: Per-Shape Transpose Bandwidth ({dtype_label}) — "
                     f"{GPU_LABEL} ({COMMIT})", fontsize=13, y=1.02)
        fig.tight_layout()
        path = os.path.join(out_dir, f"exp1b_transpose_{dtype_label}.pdf")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_batch_sweep(f32_csv, f64_csv, out_dir):
    """Batch sweep: speedup vs batch size, f32+f64 on one plot, fixed vs mixed."""
    # Collect data from both dtypes
    series = {}  # key: (dtype, mode) → lists of (num_blocks, speedup)
    for dtype_label, csv_path in [("f32", f32_csv), ("f64", f64_csv)]:
        if not csv_path or not os.path.exists(csv_path):
            continue
        rows = parse_transpose_csv(csv_path)
        for r in rows:
            if r.get("section") != "batch_sweep":
                continue
            mode = r.get("mode", "")
            su = r.get("speedup", 0.0)
            nb = r.get("num_blocks", 0)
            if su <= 0 or nb <= 0:
                continue
            key = (dtype_label, mode)
            series.setdefault(key, ([], []))
            series[key][0].append(nb)
            series[key][1].append(su)

    if not series:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    style_map = {
        ("f32", "fixed"): dict(color="#1976d2", linestyle="-",  marker="o", label="f32 fixed"),
        ("f32", "mixed"): dict(color="#1976d2", linestyle="--", marker="s", label="f32 mixed"),
        ("f64", "fixed"): dict(color="#d32f2f", linestyle="-",  marker="o", label="f64 fixed"),
        ("f64", "mixed"): dict(color="#d32f2f", linestyle="--", marker="s", label="f64 mixed"),
    }

    for key, (xs, ys) in sorted(series.items()):
        sty = style_map.get(key, {})
        ax.plot(xs, ys, markersize=6, linewidth=1.8, **sty)

    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 50, 100, 500, 1000])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Batch size (number of blocks)", fontsize=11)
    ax.set_ylabel("Speedup (cuTENSOR / Hypertile)", fontsize=11)
    ax.axhline(y=1.0, color="#888", linestyle=":", linewidth=0.8)
    ax.set_title(f"Transpose Speedup vs Batch Size (rank=5, ~1 GB total)\n"
                 f"{GPU_LABEL} ({COMMIT})", fontsize=12, pad=10)
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, "batch_sweep_transpose.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Summary tables ─────────────────────────────────────────────────────

def print_synthetic_summary(groups):
    print(f"\n=== Synthetic Benchmark Summary ({COMMIT}, {GPU_LABEL}, f32) ===\n")
    print(f"{'Config':<45} {'Pts':>4} {'M/N equiv':>14} {'Pairs':>18} "
          f"{'Kern su range':>15} {'Gmean':>7} {'Errs':>6}")
    print("-" * 118)

    for config_name in sorted(groups.keys()):
        pts = groups[config_name]
        gd = [r for r in pts if r["status"] == "PASS" and r["exec_ms"] > 0]
        errs = sum(1 for r in pts if r["status"] in ("ERROR", "FAIL"))
        skips = sum(1 for r in pts if r["status"] == "SKIP")

        if not gd:
            e = f"{errs}" if errs else ""
            if skips: e += f"+{skips}s"
            print(f"{config_name:<45} {len(pts):>4} {'--':>14} {'--':>18} "
                  f"{'--':>15} {'--':>7} {e:>6}")
            continue

        mn_vals = [get_mn_equiv(r) for r in gd]
        mn_min = min(mn_vals)
        mn_max = max(mn_vals)
        pair_min = min(r["pairs"] for r in gd)
        pair_max = max(r["pairs"] for r in gd)
        esu = [r["exec_speedup"] for r in gd]
        gm = geomean(esu)

        e = f"{errs}" if errs else ""
        if skips: e += f"+{skips}s"

        print(f"{config_name:<45} {len(gd):>4} "
              f"{mn_min:>6.0f}-{mn_max:>5.0f}  "
              f"{pair_min:>7,}-{pair_max:>7,}  "
              f"{min(esu):>5.1f}x-{max(esu):>5.1f}x  "
              f"{gm:>5.2f}x  "
              f"{e:>6}")


def print_pipeline_summary(f32_csv, f64_csv):
    print(f"\n=== Pipeline Benchmark Summary ({COMMIT}, {GPU_LABEL}) ===")
    print("  speedup = cuTENSOR_ms / pipeline_ms (>1 = pipeline faster)\n")
    print(f"{'Case':<50} {'f32 su':>8} {'f32 pipe':>10} {'f32 ct':>8} "
          f"{'f64 su':>8} {'f64 pipe':>10} {'f64 ct':>8}")
    print("-" * 108)

    f32 = {r["case_name"]: r for r in read_csv(f32_csv)}
    f64 = {r["case_name"]: r for r in read_csv(f64_csv)}
    f32_sus, f64_sus = [], []

    all_names = sorted(set(list(f32.keys()) + list(f64.keys())))
    for name in all_names:
        sname = name.split("_symm_")[-1] if "_symm_" in name else name[-20:]
        cat = "can" if "canonical" in name else ("int" if "internal" in name else "mat")
        display = f"[{cat}] {sname}"

        def fmt(r, collector):
            pipe = float(r.get("cutlass_ms", 0))
            ct = float(r.get("cutensor_ms", 0))
            if pipe == 0 and ct == 0:
                return "  OOM   ", "    --    ", "   --   "
            su = ct / pipe if pipe > 0 else 0
            if su > 0:
                collector.append(su)
            return f"{su:>7.2f}x", f"{pipe:>9.2f}", f"{ct:>7.2f}"

        su32, p32, c32 = fmt(f32.get(name, {}), f32_sus)
        su64, p64, c64 = fmt(f64.get(name, {}), f64_sus)
        print(f"{display:<50} {su32} {p32} {c32} {su64} {p64} {c64}")

    gm32 = geomean(f32_sus)
    gm64 = geomean(f64_sus)
    print(f"\n{'Geometric mean':<50} {gm32:>7.2f}x {'':>10} {'':>8} "
          f"{gm64:>7.2f}x")


def print_transpose_summary(f32_csv, f64_csv):
    print(f"\n=== Transpose Benchmark Summary ({COMMIT}, {GPU_LABEL}) ===")
    print("  speedup = hypertile_gbps / cutensor_gbps\n")

    for dtype_label, csv_path in [("f32", f32_csv), ("f64", f64_csv)]:
        if not os.path.exists(csv_path):
            continue
        rows = parse_transpose_csv(csv_path)
        good = [r for r in rows if r["ht_gbps"] > 0 and r["ct_gbps"] > 0]
        if not good:
            continue

        by_section = defaultdict(list)
        for r in good:
            by_section[r["section"]].append(r)

        print(f"  {dtype_label}:")
        all_su = []
        for section in ["single", "batched", "mixed"]:
            pts = by_section.get(section, [])
            if not pts:
                continue
            sus = [r["speedup"] for r in pts]
            all_su.extend(sus)
            ht_bw = [r["ht_gbps"] for r in pts]
            ct_bw = [r["ct_gbps"] for r in pts]
            gm = geomean(sus)
            print(f"    {section:<8}: {len(pts):>3} cases, "
                  f"speedup {min(sus):.2f}x-{max(sus):.2f}x (gmean={gm:.2f}x), "
                  f"HT {min(ht_bw):.0f}-{max(ht_bw):.0f} GB/s, "
                  f"CT {min(ct_bw):.0f}-{max(ct_bw):.0f} GB/s")
        if all_su:
            print(f"    {'overall':<8}: gmean={geomean(all_su):.2f}x")
        print()


# ── Tier distribution (EXP-4) ─────────────────────────────────────────

def plot_tier_distribution(f32_csv, f64_csv, out_dir):
    """Stacked bar chart showing FLOPs fraction per tier for each case."""
    for dtype_label, csv_path in [("f32", f32_csv), ("f64", f64_csv)]:
        if not csv_path or not os.path.exists(csv_path):
            continue
        rows = read_csv(csv_path)
        if not rows:
            continue

        # Filter to rows with tier flop data
        cases = []
        for r in rows:
            lf = float(r.get("large_flops_gflop", 0))
            mf = float(r.get("medium_flops_gflop", 0))
            sf = float(r.get("small_flops_gflop", 0))
            total = lf + mf + sf
            if total <= 0:
                continue
            name = r.get("case_name", "")
            cat = "can" if "canonical" in name else ("int" if "internal" in name else "mat")
            parts = name.split("_symm_")
            suffix = parts[1] if len(parts) == 2 else name[-15:]
            cases.append({
                "name": name,
                "label": f"{cat}/{suffix}",
                "cat": cat,
                "large_frac": lf / total,
                "medium_frac": mf / total,
                "small_frac": sf / total,
                "total_gflop": total,
            })

        if not cases:
            continue

        # Sort by category then name
        cat_order = {"can": 0, "int": 1, "mat": 2}
        cases.sort(key=lambda c: (cat_order.get(c["cat"], 9), c["name"]))

        n = len(cases)
        fig, ax = plt.subplots(figsize=(max(14, n * 0.35), 6))
        x = list(range(n))

        large_vals = [c["large_frac"] for c in cases]
        medium_vals = [c["medium_frac"] for c in cases]
        small_vals = [c["small_frac"] for c in cases]

        ax.bar(x, large_vals, color="#1976d2", label="Large (CUTLASS)", alpha=0.85)
        ax.bar(x, medium_vals, bottom=large_vals, color="#ff9800",
               label="Medium (Volume)", alpha=0.85)
        ax.bar(x, small_vals,
               bottom=[l + m for l, m in zip(large_vals, medium_vals)],
               color="#4caf50", label="Small (Warp)", alpha=0.85)

        labels = [c["label"] for c in cases]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel("FLOPs Fraction", fontsize=11)
        ax.set_title(f"Tier Distribution by FLOPs ({dtype_label}) — "
                     f"{GPU_LABEL} ({COMMIT})", fontsize=13, pad=15)
        ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        path = os.path.join(out_dir, f"tier_distribution_{dtype_label}.pdf")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ── Case study (EXP-3) ───────────────────────────────────────────────

def plot_case_study(f32_csv, f64_csv, out_dir):
    """Grouped bar chart: speedup by symmetry x size, colored by layout."""
    LAYOUT_COLORS = {
        "canonical": "#1976d2",
        "int_perm": "#ff9800",
        "mat_perm": "#4caf50",
    }
    LAYOUT_LABELS = {
        "canonical": "Canonical",
        "int_perm": "Internal permuted",
        "mat_perm": "Matrix permuted",
    }

    for dtype_label, csv_path in [("f32", f32_csv), ("f64", f64_csv)]:
        if not csv_path or not os.path.exists(csv_path):
            continue
        rows = read_csv(csv_path)
        if not rows:
            continue

        # Filter to rows with layout info
        cases = []
        for r in rows:
            dl = r.get("data_layout", "")
            sym = r.get("symmetry", "")
            sz = r.get("size", "")
            ct = float(r.get("cutensor_ms", 0))
            pipe = float(r.get("cutlass_ms", 0))
            if not dl or not sym or not sz or ct <= 0 or pipe <= 0:
                continue
            cases.append({
                "sym_size": f"{sym}_{sz}",
                "layout": dl,
                "speedup": ct / pipe,
            })

        if not cases:
            continue

        # Determine unique sym_size labels and layouts
        sym_sizes = sorted(set(c["sym_size"] for c in cases))
        layouts = [l for l in ["canonical", "int_perm", "mat_perm"]
                   if any(c["layout"] == l for c in cases)]

        # Build lookup
        lookup = {}
        for c in cases:
            lookup[(c["sym_size"], c["layout"])] = c["speedup"]

        n = len(sym_sizes)
        num_layouts = len(layouts)
        w = 0.8 / max(num_layouts, 1)
        fig, ax = plt.subplots(figsize=(max(12, n * 0.6), 6))

        for j, layout in enumerate(layouts):
            offsets = [i + (j - (num_layouts - 1) / 2) * w for i in range(n)]
            vals = [lookup.get((ss, layout), 0) for ss in sym_sizes]
            ax.bar(offsets, vals, w, label=LAYOUT_LABELS.get(layout, layout),
                   color=LAYOUT_COLORS.get(layout, "#999"), alpha=0.85,
                   edgecolor="white", linewidth=0.3)

        ax.axhline(y=1.0, color="#888", linestyle=":", linewidth=0.8)
        ax.set_xticks(range(n))
        ax.set_xticklabels(sym_sizes, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Speedup (cuTENSOR / Pipeline)", fontsize=11)
        ax.set_title(f"Pipeline Speedup: Case Study ({dtype_label}) — "
                     f"{GPU_LABEL} ({COMMIT})", fontsize=13, pad=15)
        ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        path = os.path.join(out_dir, f"case_study_{dtype_label}.pdf")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ── Pipeline heatmap ──────────────────────────────────────────────────

SYMMETRY_ORDER = ["nfparity", "nf", "sznf", "kysznf"]  # coarse → fine
SIZE_ORDER = ["E1", "E2", "E3", "S1", "S2", "S3"]
LAYOUT_LABELS_HEAT = {
    "canonical": "Canonical",
    "int_perm": "Internal perm.",
    "mat_perm": "Matrix perm.",
}

HEAT_VMIN, HEAT_VMAX = 0.5, 4.0  # colour-scale clamp


def plot_pipeline_heatmap(f32_csv, f64_csv, out_dir):
    """2D heatmap: speedup by symmetry (y) x size (x), one panel per formulation."""
    for dtype_label, csv_path in [("f32", f32_csv), ("f64", f64_csv)]:
        if not csv_path or not os.path.exists(csv_path):
            continue
        rows = read_csv(csv_path)
        if not rows:
            continue

        # Build lookup: (layout, symmetry, size) → speedup
        lookup = {}
        for r in rows:
            dl = r.get("data_layout", "")
            sym = r.get("symmetry", "")
            sz = r.get("size", "")
            ct = float(r.get("cutensor_ms", 0))
            pipe = float(r.get("cutlass_ms", 0))
            if not dl or not sym or not sz or pipe <= 0:
                continue
            su = ct / pipe if ct > 0 else 0
            lookup[(dl, sym, sz)] = su

        # Determine which sizes/symmetries are present
        syms = [s for s in SYMMETRY_ORDER if any(k[1] == s for k in lookup)]
        sizes = [s for s in SIZE_ORDER if any(k[2] == s for k in lookup)]
        layouts = [l for l in ["canonical", "int_perm", "mat_perm"]
                   if any(k[0] == l for k in lookup)]

        if not syms or not sizes or not layouts:
            continue

        n_layouts = len(layouts)
        pw = 0.72 * len(sizes)   # per-panel width from cell count
        cbar_w = 0.35            # space for colorbar
        fig_w = pw * n_layouts + cbar_w + 0.55
        fig_h = 0.72 * len(syms) + 0.95

        fig, axes = plt.subplots(
            1, n_layouts, figsize=(fig_w, fig_h), squeeze=False,
            gridspec_kw={"wspace": 0.05,
                         "left": 0.08, "right": 1 - cbar_w / fig_w,
                         "bottom": 0.16, "top": 0.84})

        norm = TwoSlopeNorm(vcenter=1.0, vmin=HEAT_VMIN, vmax=HEAT_VMAX)
        cmap = plt.cm.RdYlGn

        im = None
        for idx, layout in enumerate(layouts):
            ax = axes[0, idx]
            data = np.full((len(syms), len(sizes)), np.nan)
            for i, sym in enumerate(syms):
                for j, sz in enumerate(sizes):
                    val = lookup.get((layout, sym, sz))
                    if val is not None and val > 0:
                        data[i, j] = val

            # Clamp for colour mapping only; annotations show true value
            data_clamped = np.clip(data, HEAT_VMIN, HEAT_VMAX)
            im = ax.imshow(data_clamped, cmap=cmap, norm=norm, aspect="equal",
                           interpolation="nearest")

            for i in range(len(syms)):
                for j in range(len(sizes)):
                    val = data[i, j]
                    if np.isnan(val):
                        ax.text(j, i, "\u2014", ha="center", va="center",
                                fontsize=11, color="#999")
                    else:
                        # high-contrast text
                        mapped = norm(np.clip(val, HEAT_VMIN, HEAT_VMAX))
                        color = "white" if mapped < 0.25 or mapped > 0.85 else "black"
                        txt = f"{val:.1f}" if val >= 10 else f"{val:.2f}"
                        ax.text(j, i, txt, ha="center", va="center",
                                fontsize=11, fontweight="bold", color=color)

            ax.set_xticks(range(len(sizes)))
            ax.set_xticklabels(sizes, fontsize=12)
            ax.set_yticks(range(len(syms)))
            ax.set_yticklabels(syms if idx == 0 else [], fontsize=12)
            if idx == 0:
                ax.set_ylabel("Symmetry", fontsize=13, labelpad=6)
            ax.set_xlabel("Size class", fontsize=13)
            ax.set_title(LAYOUT_LABELS_HEAT.get(layout, layout),
                         fontsize=13, pad=8)
            ax.tick_params(length=0)  # hide tick marks

        dtype_pretty = dtype_label.upper().replace("F", "FP")
        fig.suptitle(f"TIDES / cuTENSOR speedup ({dtype_pretty})",
                     fontsize=14, fontweight="bold", y=0.97)

        # Colorbar
        cbar_left = 1 - cbar_w / fig_w + 0.02
        cbar_ax = fig.add_axes([cbar_left, 0.16, 0.015, 0.68])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Speedup", fontsize=12, labelpad=8)
        cbar.ax.tick_params(labelsize=11)
        cbar.set_ticks([0.5, 1.0, 2.0, 3.0, 4.0])
        cbar.set_ticklabels(["0.5\u00d7", "1\u00d7", "2\u00d7", "3\u00d7", "4\u00d7"])

        path = os.path.join(out_dir, f"pipeline_heatmap_{dtype_label}.pdf")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    global COMMIT, GPU_LABEL

    parser = argparse.ArgumentParser(
        description="Plot benchmark results (auto-discovers latest runs)")
    parser.add_argument("--run", help="Commit prefix to match across all categories "
                        "(e.g. '5e139c5')")
    parser.add_argument("--out-dir", help="Output directory for plots")
    parser.add_argument("--synth-dir", help="Synthetic results directory")
    parser.add_argument("--pipe-f32", help="Pipeline f32 results directory")
    parser.add_argument("--pipe-f64", help="Pipeline f64 results directory")
    parser.add_argument("--trans-f32", help="Transpose f32 results directory")
    parser.add_argument("--trans-f64", help="Transpose f64 results directory")
    parser.add_argument("--commit", help="Override commit label")
    parser.add_argument("--gpu", help="Override GPU label")
    parser.add_argument("--cherry-pick",
                        help="Filter pipeline/case-study plots: comma-separated "
                             "case name substrings or @filename")
    args = parser.parse_args()

    run = args.run  # commit prefix filter (None = latest)

    # ── Resolve directories ──
    synth_dir = args.synth_dir or _latest_dir("synthetic", run)

    # When no --run given, infer commit from synth_dir so all categories
    # auto-discover from the same commit (prevents mixing e.g. latest
    # synthetic from commit A with latest pipeline from commit B).
    if not run and synth_dir:
        synth_commit, _ = _read_env(synth_dir)
        if synth_commit:
            run = synth_commit

    pipe_f32_dir = args.pipe_f32
    pipe_f64_dir = args.pipe_f64
    if not pipe_f32_dir or not pipe_f64_dir:
        auto_f32, auto_f64 = _find_dirs_by_dtype("pipeline", run)
        pipe_f32_dir = pipe_f32_dir or auto_f32
        pipe_f64_dir = pipe_f64_dir or auto_f64

    trans_f32_dir = args.trans_f32
    trans_f64_dir = args.trans_f64
    if not trans_f32_dir or not trans_f64_dir:
        auto_f32, auto_f64 = _find_dirs_by_dtype("transpose", run,
                                                   exclude_flag="--batch-sweep")
        trans_f32_dir = trans_f32_dir or auto_f32
        trans_f64_dir = trans_f64_dir or auto_f64

    # Batch-sweep transpose dirs (separate from regular transpose)
    bsweep_f32_dir, bsweep_f64_dir = _find_dirs_by_dtype(
        "transpose", run, require_flag="--batch-sweep")

    # Check we have something to plot
    has_synth = synth_dir is not None
    has_pipe = pipe_f32_dir or pipe_f64_dir
    has_trans = trans_f32_dir or trans_f64_dir
    has_bsweep = bsweep_f32_dir or bsweep_f64_dir
    if not has_synth and not has_pipe and not has_trans and not has_bsweep:
        parser.error("No results found; pass explicit dirs or --run")

    # ── Commit / GPU label ──
    # Try to read env from the first available result dir
    env_dir = synth_dir or pipe_f32_dir or pipe_f64_dir or trans_f32_dir or trans_f64_dir
    auto_commit, auto_gpu = _read_env(env_dir) if env_dir else ("", "")
    COMMIT = args.commit or auto_commit or "unknown"
    GPU_LABEL = args.gpu or auto_gpu or "GPU"

    # Output dir: prefer synth plots dir, else first available result dir
    default_out = os.path.join(synth_dir, "plots") if synth_dir else \
                  os.path.join(env_dir, "plots")
    out_dir = args.out_dir or default_out
    os.makedirs(out_dir, exist_ok=True)

    print(f"Commit: {COMMIT}  GPU: {GPU_LABEL}")
    if has_synth:
        print(f"  synthetic: {synth_dir}")
    print(f"  pipeline:  f32={pipe_f32_dir}  f64={pipe_f64_dir}")
    print(f"  transpose: f32={trans_f32_dir}  f64={trans_f64_dir}")
    print(f"  batch-sweep: f32={bsweep_f32_dir}  f64={bsweep_f64_dir}")

    # ── Synthetic ──
    groups = {}
    if has_synth:
        synth_csv = os.path.join(synth_dir, "results.csv")
        if not os.path.exists(synth_csv):
            import subprocess
            parse_script = os.path.join(os.path.dirname(__file__), "parse_results.py")
            output_txt = os.path.join(synth_dir, "output.txt")
            subprocess.run(["python3", parse_script, output_txt, "--csv", "-o", synth_csv],
                           check=True)

        print("\nParsing synthetic results...")
        synth_rows = parse_synth_csv(synth_csv)
        groups = group_by_config(synth_rows)
        print(f"  {len(synth_rows)} rows, {len(groups)} configs\n")

        print_synthetic_summary(groups)

    # ── Pipeline ──
    f32_csv = os.path.join(pipe_f32_dir, "results.csv") if pipe_f32_dir else ""
    f64_csv = os.path.join(pipe_f64_dir, "results.csv") if pipe_f64_dir else ""
    if f32_csv and f64_csv and os.path.exists(f32_csv) and os.path.exists(f64_csv):
        print_pipeline_summary(f32_csv, f64_csv)

    # ── Transpose ──
    trans_f32_csv = os.path.join(trans_f32_dir, "results.csv") if trans_f32_dir else ""
    trans_f64_csv = os.path.join(trans_f64_dir, "results.csv") if trans_f64_dir else ""
    print_transpose_summary(trans_f32_csv, trans_f64_csv)

    print(f"\nGenerating plots in {out_dir}/...\n")

    if groups:
        plot_fig1_by_tier(groups, out_dir)
        plot_fig2_combined(groups, out_dir)
        plot_setup_scaling(groups, out_dir)
        plot_fig1_setup_vs_mn(groups, out_dir)

    picks = _load_cherry_pick(args.cherry_pick)

    if f32_csv and os.path.exists(f32_csv):
        plot_pipeline(f32_csv, f64_csv, out_dir, cherry_pick=picks)
        plot_pipeline_heatmap(f32_csv, f64_csv, out_dir)
        plot_tier_distribution(f32_csv, f64_csv, out_dir)
        plot_case_study(f32_csv, f64_csv, out_dir)

    if trans_f32_csv or trans_f64_csv:
        plot_transpose(trans_f32_csv, trans_f64_csv, out_dir)
        plot_exp1b(trans_f32_csv, trans_f64_csv, out_dir)

    bsweep_f32_csv = os.path.join(bsweep_f32_dir, "results.csv") if bsweep_f32_dir else ""
    bsweep_f64_csv = os.path.join(bsweep_f64_dir, "results.csv") if bsweep_f64_dir else ""
    if bsweep_f32_csv or bsweep_f64_csv:
        plot_batch_sweep(bsweep_f32_csv, bsweep_f64_csv, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
