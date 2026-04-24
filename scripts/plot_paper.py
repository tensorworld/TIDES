#!/usr/bin/env python3
"""Generate publication-quality figures for the SC paper.

Reads CSV data from pipeline and transpose benchmarks and produces compact
PDF figures sized for a two-column IEEEtran layout (~3.5 in column width,
~7.0 in full page width).

Usage:
    python3 scripts/plot_paper.py \
        --pipe-f64 results/pipeline/DIR/results.csv \
        --pipe-f32 results/pipeline/DIR/results.csv \
        --trans-f64 results/transpose/DIR/results.csv \
        --out-dir results/paper_figures/
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_CUTENSOR = "#e07050"
COLOR_HYPERTILE = "#4caf50"

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def read_csv_rows(path):
    """Read a CSV file and return a list of dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


def parse_transpose_csv(path):
    """Read transpose CSV, coercing numeric fields."""
    rows = read_csv_rows(path)
    for r in rows:
        for k in ("cutensor_ms", "hypertile_ms", "speedup",
                   "ht_gbps", "ct_gbps"):
            if k in r:
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    r[k] = 0.0
        for k in ("num_blocks",):
            if k in r:
                try:
                    r[k] = int(r[k])
                except (ValueError, TypeError):
                    r[k] = 0
    return rows


# ---------------------------------------------------------------------------
# Figure 1: Pipeline speedup heatmap
# ---------------------------------------------------------------------------

SYMMETRY_ORDER = ["nfparity", "nf", "sznf", "kysznf"]
SIZE_ORDER = ["E1", "E2", "S1", "S2", "S3"]
LAYOUT_ORDER = ["canonical", "int_perm"]
LAYOUT_LABELS = {"canonical": "Canonical", "int_perm": "Internal perm."}
HEAT_VMIN, HEAT_VMAX = 0.5, 4.0


def plot_pipeline_heatmap(csv_path, dtype_label, out_dir):
    """Generate a 2-panel heatmap (canonical, int_perm) for one dtype."""
    rows = read_csv_rows(csv_path)
    if not rows:
        return

    # Build lookup: (layout, symmetry, size) -> speedup
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

    # Filter to present data in fixed order
    syms = [s for s in SYMMETRY_ORDER if any(k[1] == s for k in lookup)]
    sizes = [s for s in SIZE_ORDER if any(k[2] == s for k in lookup)]
    layouts = [l for l in LAYOUT_ORDER if any(k[0] == l for k in lookup)]

    if not syms or not sizes or not layouts:
        return

    n_layouts = len(layouts)
    fig, axes = plt.subplots(
        1, n_layouts, figsize=(5.5, 2.2), squeeze=False,
        gridspec_kw={"wspace": 0.05,
                     "left": 0.10, "right": 0.88,
                     "bottom": 0.18, "top": 0.82})

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

        data_clamped = np.clip(data, HEAT_VMIN, HEAT_VMAX)
        im = ax.imshow(data_clamped, cmap=cmap, norm=norm, aspect="equal",
                       interpolation="nearest")

        # Cell annotations
        for i in range(len(syms)):
            for j in range(len(sizes)):
                val = data[i, j]
                if np.isnan(val):
                    ax.text(j, i, "\u2014", ha="center", va="center",
                            fontsize=9, color="#999")
                else:
                    mapped = norm(np.clip(val, HEAT_VMIN, HEAT_VMAX))
                    color = "white" if mapped < 0.25 or mapped > 0.85 else "black"
                    txt = f"{val:.1f}" if val >= 10 else f"{val:.2f}"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=9, fontweight="bold", color=color)

        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels(sizes, fontsize=8)
        ax.set_yticks(range(len(syms)))
        ax.set_yticklabels(syms if idx == 0 else [], fontsize=8)
        if idx == 0:
            ax.set_ylabel("Symmetry", fontsize=9, labelpad=4)
        ax.set_xlabel("Size class", fontsize=9)
        ax.set_title(LAYOUT_LABELS.get(layout, layout), fontsize=9, pad=6)
        ax.tick_params(length=0)

    dtype_pretty = dtype_label.upper().replace("F", "FP")
    fig.suptitle(f"TIDES / cuTENSOR speedup ({dtype_pretty})",
                 fontsize=10, fontweight="bold", y=0.96)

    # Colorbar
    cbar_ax = fig.add_axes([0.90, 0.18, 0.015, 0.64])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0.5, 1.0, 2.0, 3.0, 4.0])
    cbar.set_ticklabels(["0.5\u00d7", "1\u00d7", "2\u00d7", "3\u00d7", "4\u00d7"])

    path = os.path.join(out_dir, f"pipeline_heatmap_{dtype_label}.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _build_lookup(csv_path):
    """Build (layout, symmetry, size) -> speedup lookup from pipeline CSV."""
    rows = read_csv_rows(csv_path)
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
    return lookup


def plot_pipeline_heatmap_stacked(csv_f64, csv_f32, out_dir):
    """Generate a stacked (a) FP64 / (b) FP32 heatmap with shared colorbar."""
    lookup64 = _build_lookup(csv_f64)
    lookup32 = _build_lookup(csv_f32)

    if not lookup64 or not lookup32:
        return

    syms = [s for s in SYMMETRY_ORDER
            if any(k[1] == s for k in lookup64)]
    sizes = [s for s in SIZE_ORDER
             if any(k[2] == s for k in lookup64)]
    layouts = [l for l in LAYOUT_ORDER
               if any(k[0] == l for k in lookup64)]

    if not syms or not sizes or not layouts:
        return

    n_layouts = len(layouts)
    fig, axes = plt.subplots(
        2, n_layouts, figsize=(5.5, 3.2), squeeze=False,
        gridspec_kw={"wspace": 0.1, "hspace": 0.1,
                     "left": 0.08, "right": 0.88,
                     "bottom": 0.10, "top": 0.90})

    norm = TwoSlopeNorm(vcenter=1.0, vmin=HEAT_VMIN, vmax=HEAT_VMAX)
    cmap = plt.cm.RdYlGn

    im = None
    for row_idx, (lookup, dtype_label) in enumerate(
            [(lookup64, "FP64"), (lookup32, "FP32")]):
        for col_idx, layout in enumerate(layouts):
            ax = axes[row_idx, col_idx]
            data = np.full((len(syms), len(sizes)), np.nan)
            for i, sym in enumerate(syms):
                for j, sz in enumerate(sizes):
                    val = lookup.get((layout, sym, sz))
                    if val is not None and val > 0:
                        data[i, j] = val

            data_clamped = np.clip(data, HEAT_VMIN, HEAT_VMAX)
            im = ax.imshow(data_clamped, cmap=cmap, norm=norm,
                           aspect="auto", interpolation="nearest")

            # Cell annotations
            for i in range(len(syms)):
                for j in range(len(sizes)):
                    val = data[i, j]
                    if np.isnan(val):
                        ax.text(j, i, "\u2014", ha="center", va="center",
                                fontsize=8, color="#999")
                    else:
                        mapped = norm(np.clip(val, HEAT_VMIN, HEAT_VMAX))
                        color = ("white" if mapped < 0.25 or mapped > 0.85
                                 else "black")
                        txt = (f"{val:.1f}" if val >= 10
                               else f"{val:.2f}")
                        ax.text(j, i, txt, ha="center", va="center",
                                fontsize=8, fontweight="bold", color=color)

            ax.set_xticks(range(len(sizes)))
            show_x = (row_idx == 1)  # x labels only on bottom row
            ax.set_xticklabels(sizes if show_x else [], fontsize=8)
            ax.set_yticks(range(len(syms)))
            ax.set_yticklabels(syms if col_idx == 0 else [], fontsize=8)
            ax.tick_params(length=0)

            # Top row: layout titles
            if row_idx == 0:
                ax.set_title(LAYOUT_LABELS.get(layout, layout),
                             fontsize=9, pad=4)

        # Row label: (a) FP64 / (b) FP32
        panel = "a" if row_idx == 0 else "b"
        axes[row_idx, 0].set_ylabel(f"({panel}) {dtype_label}",
                                     fontsize=9, labelpad=8)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.90, 0.10, 0.015, 0.80])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([0.5, 1.0, 2.0, 3.0, 4.0])
    cbar.set_ticklabels(["0.5\u00d7", "1\u00d7", "2\u00d7",
                          "3\u00d7", "4\u00d7"])

    path = os.path.join(out_dir, "pipeline_heatmap_combined.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2: Batch sweep transpose speedup
# ---------------------------------------------------------------------------

def plot_batch_sweep(trans_csvs, out_dir):
    """Line plot: speedup vs batch size.

    trans_csvs: list of (dtype_label, csv_path) tuples.
    If only one dtype provided, show 2 series; if both, show 4.
    """
    series = {}  # (dtype, mode) -> (xs, ys)
    for dtype_label, csv_path in trans_csvs:
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

    fig, ax = plt.subplots(figsize=(3.3, 2.5))

    style_map = {
        ("f32", "fixed"): dict(color="#1976d2", ls="-",  marker="o", label="FP32 batched"),
        ("f32", "mixed"): dict(color="#1976d2", ls="--", marker="s", label="FP32 grouped"),
        ("f64", "fixed"): dict(color="#d32f2f", ls="-",  marker="o", label="FP64 batched"),
        ("f64", "mixed"): dict(color="#d32f2f", ls="--", marker="s", label="FP64 grouped"),
    }

    for key in sorted(series.keys()):
        xs, ys = series[key]
        sty = style_map.get(key, {})
        ax.plot(xs, ys, markersize=4, linewidth=1.2, **sty)

    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 50, 100, 500, 1000])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("Batch size (number of blocks)", fontsize=8)
    ax.set_ylabel("Speedup (cuTENSOR / Hypertile)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.axhline(y=1.0, color="#888", linestyle=":", linewidth=0.8)
    ax.set_title("Transpose speedup vs. batch size", fontsize=9, pad=6)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9,
              handlelength=1.5, borderpad=0.3, labelspacing=0.3)
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)

    path = os.path.join(out_dir, "batch_sweep.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3: EXP-1b per-shape transpose bandwidth
# ---------------------------------------------------------------------------

def _short_perm(perm_str):
    """'1,0,2,3' -> 'p1023'."""
    return "p" + perm_str.replace(",", "").replace(" ", "")


def _compact_shape(shape_str):
    """'373x6x200x2' -> '373x6x200x2' (no change, but could abbreviate)."""
    return shape_str


def plot_exp1b(csv_path, out_dir):
    """Single-axis bar chart with (a) batched per-shape and (b) grouped real
    perms separated by a vertical line.  Shared y-axis.

    Panel (a): 3 shapes x 3 perms = 9 bars (4D Hypertile-wins, 6D mixed,
    6D cuTENSOR-wins).
    Panel (b): E2 + S2, operand A, 4 symmetries each = 8 bars, ordered
    kysznf -> sznf -> nf -> nfparity within each system.
    """
    rows = parse_transpose_csv(csv_path)
    if not rows:
        return

    # ---- Panel (a): select 3 shapes, 3 perms each ----
    SHAPES_A = ["373x6x200x2", "65x15x16x6x6x2", "802x801x6x6x2x1"]
    rows_a_all = [r for r in rows
                  if r.get("section") == "exp1b_a" and r.get("ht_gbps", 0) > 0]
    rows_a = []
    for shape in SHAPES_A:
        group = [r for r in rows_a_all if r.get("shape") == shape]
        rows_a.extend(group[:3])  # first 3 perms per shape

    # ---- Panel (b): E2 + S2, operand A, 4 symmetries ----
    SYM_ORDER_B = ["kysznf", "sznf", "nf", "nfparity"]
    SYSTEMS_B = ["E2", "S2"]
    rows_b_all = [r for r in rows
                  if r.get("section") == "exp1b_b" and r.get("ht_gbps", 0) > 0
                  and r.get("operand", "") == "A"]

    def _get_sym(case_name):
        for sym in SYM_ORDER_B:
            if f"_symm_{sym}_" in case_name:
                return sym
        return ""

    def _get_system(case_name):
        for sys in SYSTEMS_B:
            if f"_{sys}" in case_name:
                return sys
        return ""

    rows_b = []
    for sys in SYSTEMS_B:
        for sym in SYM_ORDER_B:
            match = [r for r in rows_b_all
                     if _get_sym(r["case_name"]) == sym
                     and _get_system(r["case_name"]) == sys]
            if match:
                rows_b.append(match[0])

    if not rows_a and not rows_b:
        return

    n_a = len(rows_a)
    n_b = len(rows_b)
    gap = 0.5  # visual gap between panels

    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    fig.subplots_adjust(bottom=0.32, top=0.88)
    w = 0.35
    annot_fs = 5.5   # speedup annotation font size (both panels)
    label_fs = 6.5   # x-tick label font size (both panels)
    group_fs = 7   # group label font size (shape / E2-S2)
    group_off = -45   # points below axes bottom (clears rotated tick labels)

    # ---- Panel (a): batched per-shape (positions 0..n_a-1) ----
    x_a = np.arange(n_a, dtype=float)
    # Compact perm labels: "p1023"
    labels_a = [_short_perm(r['perm']) for r in rows_a]
    ht_a = [r["ht_gbps"] for r in rows_a]
    ct_a = [r["ct_gbps"] for r in rows_a]
    su_a = [r["speedup"] for r in rows_a]

    ax.bar(x_a - w / 2, ct_a, w, color=COLOR_CUTENSOR, alpha=0.85,
           edgecolor="white", linewidth=0.3, label="cuTENSOR")
    ax.bar(x_a + w / 2, ht_a, w, color=COLOR_HYPERTILE, alpha=0.85,
           edgecolor="white", linewidth=0.3, label="Hypertile")

    # Compute y_max across both panels
    all_bw = ht_a + ct_a
    if rows_b:
        all_bw += [r["ht_gbps"] for r in rows_b] + [r["ct_gbps"] for r in rows_b]
    y_max = max(all_bw)
    annot_off = y_max * 0.02

    # Speedup annotations for panel (a)
    for i, su in enumerate(su_a):
        top = max(ht_a[i], ct_a[i])
        color = "#2e7d32" if su >= 1.0 else "#c62828"
        txt = f"{su:.0f}\u00d7" if su >= 10 else f"{su:.1f}\u00d7"
        ax.text(x_a[i], top + annot_off, txt, ha="center", va="bottom",
                fontsize=annot_fs, color=color, fontweight="bold")

    # Shape-group separators
    shapes_seen = []
    group_starts = []
    for i, r in enumerate(rows_a):
        s = r["shape"]
        if not shapes_seen or shapes_seen[-1] != s:
            if shapes_seen:
                ax.axvline(x=i - 0.5, color="#555", linestyle="-",
                           linewidth=0.6)
            shapes_seen.append(s)
            group_starts.append(i)
    group_starts.append(n_a)

    # Short shape labels (S1, S2, S3) — full shapes go in caption
    SHAPE_BLOCKS = {SHAPES_A[0]: 73, SHAPES_A[1]: 58, SHAPES_A[2]: 1}
    SHAPE_LABELS = {
        s: f"T{i+1}\n({SHAPE_BLOCKS[s]} block{'s' if SHAPE_BLOCKS[s] != 1 else ''})"
        for i, s in enumerate(SHAPES_A)
    }
    for g in range(len(shapes_seen)):
        mid = (group_starts[g] + group_starts[g + 1] - 1) / 2.0
        slabel = SHAPE_LABELS.get(shapes_seen[g], shapes_seen[g])
        ax.annotate(slabel, xy=(mid, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, group_off), textcoords="offset points",
                    ha="center", va="top", fontsize=group_fs, color="#333",
                    fontweight="bold", annotation_clip=False)

    # ---- Separator between (a) and (b) ----
    sep_x = n_a - 0.5 + gap / 2
    ax.axvline(x=sep_x, color="#222", linestyle="-", linewidth=1.2)

    # ---- Panel (b): grouped E2 + S2 (positions start after gap) ----
    x_b = np.arange(n_b, dtype=float) + n_a + gap

    labels_b = []
    for r in rows_b:
        sym = _get_sym(r["case_name"])
        nb = r.get("num_blocks", 0)
        labels_b.append(f"{sym}({nb})")

    ht_b = [r["ht_gbps"] for r in rows_b]
    ct_b = [r["ct_gbps"] for r in rows_b]
    su_b = [r["speedup"] for r in rows_b]

    ax.bar(x_b - w / 2, ct_b, w, color=COLOR_CUTENSOR, alpha=0.85,
           edgecolor="white", linewidth=0.3)
    ax.bar(x_b + w / 2, ht_b, w, color=COLOR_HYPERTILE, alpha=0.85,
           edgecolor="white", linewidth=0.3)

    # Speedup annotations for panel (b) — same style as (a)
    for i, su in enumerate(su_b):
        top = max(ht_b[i], ct_b[i])
        color = "#2e7d32" if su >= 1.0 else "#c62828"
        txt = f"{su:.0f}\u00d7" if su >= 10 else f"{su:.1f}\u00d7"
        ax.text(x_b[i], top + annot_off, txt, ha="center", va="bottom",
                fontsize=annot_fs, color=color, fontweight="bold")

    # Thin separator between E2 and S2 groups in panel (b)
    n_e2 = len(SYM_ORDER_B)  # first 4 bars are E2
    if n_b > n_e2:
        sep_b = x_b[n_e2] - 0.5
        ax.axvline(x=sep_b, color="#555", linestyle="-", linewidth=0.6)
        # Group labels — same style as shape labels in panel (a)
        mid_e2 = (x_b[0] + x_b[n_e2 - 1]) / 2.0
        mid_s2 = (x_b[n_e2] + x_b[-1]) / 2.0
        ax.annotate("E2", xy=(mid_e2, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, group_off), textcoords="offset points",
                    ha="center", va="top", fontsize=group_fs, color="#333",
                    fontweight="bold", annotation_clip=False)
        ax.annotate("S2", xy=(mid_s2, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, group_off), textcoords="offset points",
                    ha="center", va="top", fontsize=group_fs, color="#333",
                    fontweight="bold", annotation_clip=False)

    # ---- Axis formatting ----
    all_x = list(x_a) + list(x_b)
    all_labels = labels_a + labels_b
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=60, ha="right", fontsize=label_fs,
                       rotation_mode="anchor")
    # Convert y-axis to TB/s; label at top to save horizontal space
    ax.set_yticks([0, 1000, 2000, 3000])
    ax.set_yticklabels(["0", "1", "2", "3"])
    ax.set_ylim(0, y_max * 1.22)
    ax.text(0, 1.02, "TB/s", ha="center", va="bottom", fontsize=7,
            transform=ax.transAxes)
    ax.set_xlim(-0.6, x_b[-1] + 0.6 if len(x_b) else n_a - 0.4)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=label_fs)

    # Section titles — axes-fraction coordinates to avoid overlap
    # Compute panel centers as fractions of the x-axis range
    xlim = ax.get_xlim()
    xspan = xlim[1] - xlim[0]
    frac_a = ((n_a - 1) / 2.0 - xlim[0]) / xspan
    frac_b = (((x_b[0] + x_b[-1]) / 2.0 - xlim[0]) / xspan) if len(x_b) else 0
    ax.text(frac_a, 1.03, "(a) Batched, single shape",
            ha="center", va="bottom", fontsize=8,
            transform=ax.transAxes)
    if rows_b:
        ax.text(frac_b, 1.03, "(b) Grouped, mixed shapes",
                ha="center", va="bottom", fontsize=8,
                transform=ax.transAxes)

    path = os.path.join(out_dir, "exp1b_transpose.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for the SC paper")
    parser.add_argument("--pipe-f64",
                        help="Pipeline f64 results CSV")
    parser.add_argument("--pipe-f32",
                        help="Pipeline f32 results CSV")
    parser.add_argument("--trans-f64",
                        help="Transpose f64 results CSV (exp1b + batch sweep)")
    parser.add_argument("--trans-f32",
                        help="Transpose f32 results CSV (batch sweep only)")
    parser.add_argument("--sweep-f64",
                        help="Batch sweep f64 results CSV (if separate from --trans-f64)")
    parser.add_argument("--sweep-f32",
                        help="Batch sweep f32 results CSV (if separate from --trans-f32)")
    parser.add_argument("--out-dir", default="results/paper_figures/",
                        help="Output directory for PDFs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Figure 1: Pipeline heatmap ──
    if args.pipe_f64:
        print("Generating pipeline heatmap (FP64)...")
        plot_pipeline_heatmap(args.pipe_f64, "f64", args.out_dir)
    if args.pipe_f32:
        print("Generating pipeline heatmap (FP32)...")
        plot_pipeline_heatmap(args.pipe_f32, "f32", args.out_dir)
    if args.pipe_f64 and args.pipe_f32:
        print("Generating stacked pipeline heatmap (FP64 + FP32)...")
        plot_pipeline_heatmap_stacked(args.pipe_f64, args.pipe_f32,
                                      args.out_dir)

    # ── Figure 2: Batch sweep ──
    sweep_csvs = []
    # Prefer dedicated sweep CSVs; fall back to trans CSVs
    if args.sweep_f64:
        sweep_csvs.append(("f64", args.sweep_f64))
    elif args.trans_f64:
        sweep_csvs.append(("f64", args.trans_f64))
    if args.sweep_f32:
        sweep_csvs.append(("f32", args.sweep_f32))
    elif args.trans_f32:
        sweep_csvs.append(("f32", args.trans_f32))
    if sweep_csvs:
        print("Generating batch sweep...")
        plot_batch_sweep(sweep_csvs, args.out_dir)

    # ── Figure 3: EXP-1b (f64 only) ──
    if args.trans_f64:
        print("Generating EXP-1b transpose bandwidth...")
        plot_exp1b(args.trans_f64, args.out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
