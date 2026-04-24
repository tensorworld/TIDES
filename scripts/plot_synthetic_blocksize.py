#!/usr/bin/env python3
"""Generate publication-grade synthetic benchmark figure.

3 panels:
  (a) Large tier — block-size sweep, single line per dtype
  (b) Medium tier — MN budget sweep, median + min-max band (shape-ratio averaged)
  (c) Small tier — N sweep at M=2, median + min-max band (K-config averaged)

Usage:
    python3 scripts/plot_synthetic_blocksize.py \
        --f32 results/synthetic/.../results.csv \
        --f64 results/synthetic/.../results.csv \
        [-o figures/paper/synthetic_blocksize_sweep.pdf]
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from math import exp, log

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DTYPE_STYLE = {
    "f64": dict(color="#1565c0", marker="^", label="FP64"),
    "f32": dict(color="#c62828", marker="o", label="FP32"),
}

TIER_LABELS = {
    "L": "(a) Large tier",
    "M": "(b) Medium tier",
    "S": "(c) Small tier",
}

# ---------------------------------------------------------------------------
# Config classification
# ---------------------------------------------------------------------------

_RE_LARGE = re.compile(r'fig1: L-tensor .+\[uniform\]')
_RE_MEDIUM = re.compile(r'fig1: M-shape MN=(\d+) M=(\d+) N=(\d+)')
_RE_SMALL = re.compile(r'fig1: S-warp N=(\d+) K=(\w+)')


def classify_config(config):
    """Parse config name -> (tier, x_value) or None.

    Returns:
        ("L", None)            — large, x from row M/N
        ("M", mn_budget:int)   — medium, x = MN budget
        ("S", n_value:int)     — small, x = N
    """
    m = _RE_LARGE.match(config)
    if m:
        return ("L", None)
    m = _RE_MEDIUM.match(config)
    if m:
        return ("M", int(m.group(1)))
    m = _RE_SMALL.match(config)
    if m:
        return ("S", int(m.group(1)))
    return None


def is_tier_consistent(row, tier):
    if tier == "L":
        return row.get("large", 0) > 0
    elif tier == "M":
        return row.get("medium", 0) > 0
    elif tier == "S":
        return row.get("small", 0) > 0
    return True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_rows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("param", "exec_ms", "ct_kern_ms", "setup_ms", "ct_setup_ms",
                   "exec_speedup", "total_speedup"):
            try:
                r[k] = float(r[k])
            except (ValueError, TypeError, KeyError):
                pass
        for k in ("pairs", "large", "medium", "small", "M", "N", "K"):
            try:
                r[k] = int(r[k])
            except (ValueError, TypeError, KeyError):
                pass
    return rows


def load_data(csv_paths):
    """Load all data, keyed by (tier, x) -> [speedup_values].

    For Large: x = max(M, N).
    For Medium: x = MN budget (from config name).
    For Small: x = N (from config name).
    """
    data = defaultdict(list)
    for path in csv_paths:
        rows = parse_rows(path)
        for r in rows:
            config = r.get("config", "")
            result = classify_config(config)
            if result is None:
                continue
            tier, x_from_config = result
            if r.get("status") != "PASS":
                continue
            exec_ms = r.get("exec_ms", 0)
            ct_kern = r.get("ct_kern_ms", 0)
            if exec_ms <= 0 or ct_kern <= 0:
                continue
            if not is_tier_consistent(r, tier):
                continue
            if tier == "L":
                x = max(r.get("M", 0), r.get("N", 0))
                if x <= 0:
                    continue
            else:
                x = x_from_config
            speedup = ct_kern / exec_ms
            data[(tier, x)].append(speedup)
    return data


def aggregate(data):
    """For each (tier, x), compute median, min, max.

    Returns dict: tier -> sorted list of (x, median, min, max).
    """
    result = defaultdict(list)
    for (tier, x), vals in data.items():
        med = float(np.median(vals))
        lo = min(vals)
        hi = max(vals)
        result[tier].append((x, med, lo, hi))
    for key in result:
        result[key].sort()
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_line(ax, pts, color, marker, label):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, marker=marker, markersize=3.5, linewidth=1.1,
            markeredgewidth=0.4, markeredgecolor="white",
            color=color, label=label)


def plot_with_band(ax, pts, color, marker, label):
    xs = [p[0] for p in pts]
    medians = [p[1] for p in pts]
    mins = [p[2] for p in pts]
    maxs = [p[3] for p in pts]
    ax.plot(xs, medians, marker=marker, markersize=3.5, linewidth=1.1,
            markeredgewidth=0.4, markeredgecolor="white",
            color=color, label=label)
    ax.fill_between(xs, mins, maxs, color=color, alpha=0.15)


def plot_figure(f32_agg, f64_agg, out_path):
    fig, axes = plt.subplots(
        1, 3, figsize=(4.6, 1.6),
        gridspec_kw={"wspace": 0.2, "left": 0.09, "right": 0.98,
                     "bottom": 0.22, "top": 0.80})

    for ax_idx, tier in enumerate(["L", "M", "S"]):
        ax = axes[ax_idx]
        use_band = tier in ("M", "S")

        for dtype_label, agg in [("f64", f64_agg), ("f32", f32_agg)]:
            pts = agg.get(tier, [])
            if not pts:
                continue
            sty = DTYPE_STYLE[dtype_label]
            if use_band:
                plot_with_band(ax, pts, sty["color"], sty["marker"], sty["label"])
            else:
                plot_line(ax, pts, sty["color"], sty["marker"], sty["label"])

        ax.axhline(y=1.0, color="#888", linestyle=":", linewidth=0.7, alpha=0.6)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(labelsize=7, length=3, width=0.6)
        ax.set_title(TIER_LABELS[tier], fontsize=8, pad=5)
        ax.grid(True, axis="y", alpha=0.15, linewidth=0.4)
        ax.set_ylim(bottom=0)

        # x-axis labels
        if tier == "L":
            ax.set_xlabel("$M = N$", fontsize=8, labelpad=3)
        elif tier == "M":
            ax.set_xlabel(r"$M \times N$", fontsize=8, labelpad=3)
        elif tier == "S":
            ax.set_xlabel("$N$ ($M = 2$)", fontsize=8, labelpad=3)

    axes[0].set_ylabel("TIDES speedup", fontsize=8,
                       labelpad=3)

    # Shared legend
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)

    fig.legend(handles, labels, fontsize=7.5,
               loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.52, 1.05),
               handlelength=1.8, columnspacing=1.1,
               frameon=False)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic benchmark figure (publication-grade)")
    parser.add_argument("--f32", action="append", default=[],
                        help="FP32 results CSV (may specify multiple)")
    parser.add_argument("--f64", action="append", default=[],
                        help="FP64 results CSV (may specify multiple)")
    parser.add_argument("-o", "--output",
                        default="figures/paper/synthetic_blocksize_sweep.pdf",
                        help="Output PDF path")
    args = parser.parse_args()

    if not args.f32 and not args.f64:
        parser.error("Provide at least one --f32 or --f64 CSV")

    f32_data = load_data(args.f32)
    f64_data = load_data(args.f64)

    f32_agg = aggregate(f32_data)
    f64_agg = aggregate(f64_data)

    plot_figure(f32_agg, f64_agg, args.output)


if __name__ == "__main__":
    main()
