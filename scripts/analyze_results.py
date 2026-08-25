#!/usr/bin/env python3
"""
Analyze parsed pipeline benchmark results for paper tables/text.

Reads raw bench_pipeline_implementations output and computes:
  T1: Iteration speedup per (symmetry, layout)
  T2: Tier count table per symmetry
  T3: Tier FLOP summary per symmetry
  T4: Resource-ready prepare time comparison
  T5: Transfer time estimate vs setup overhead
  T6: Streaming workspace savings

Usage:
    python scripts/analyze_results.py results/.../output.txt          # all tables
    python scripts/analyze_results.py results/.../output.txt --only T1 T3
    python scripts/analyze_results.py results/.../output.txt --layout canonical int_perm
"""

import argparse
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set

# Import the pipeline parser from parse_results
sys.path.insert(0, __import__('os').path.dirname(__file__))
from parse_results import PipelineRow, parse_pipeline


def load_rows(path: str, layouts: Optional[Set[str]] = None) -> List[PipelineRow]:
    with open(path) as f:
        text = f.read()
    rows = parse_pipeline(text)
    if layouts:
        rows = [r for r in rows if r.data_layout in layouts]
    return rows


# ---------------------------------------------------------------------------
# T1: Iteration speedup — sum kernel times per (symmetry, layout), ratio
# ---------------------------------------------------------------------------

def t1_iteration_speedup(rows: List[PipelineRow]):
    """Sum cutlass_ms and cutensor_ms across E1-S3 per (symmetry, layout)."""
    groups: Dict[tuple, dict] = defaultdict(lambda: {'cutlass': 0.0, 'cutensor': 0.0, 'n': 0})
    for r in rows:
        if r.cutensor_ms <= 0:
            continue
        key = (r.symmetry, r.data_layout)
        groups[key]['cutlass'] += r.cutlass_ms
        groups[key]['cutensor'] += r.cutensor_ms
        groups[key]['n'] += 1

    print("=" * 70)
    print("T1: Iteration Speedup (sum of kernel times across E1-S3)")
    print("=" * 70)
    print(f"{'Symmetry':<12} {'Layout':<12} {'Cases':>5} {'TIDES ms':>10} {'cuTENSOR ms':>12} {'Speedup':>8}")
    print("-" * 70)

    for (sym, layout) in sorted(groups):
        g = groups[(sym, layout)]
        su = g['cutensor'] / g['cutlass'] if g['cutlass'] > 0 else 0
        print(f"{sym:<12} {layout:<12} {g['n']:>5} {g['cutlass']:>10.2f} {g['cutensor']:>12.2f} {su:>7.2f}x")

    # Overall
    total_c = sum(g['cutlass'] for g in groups.values())
    total_t = sum(g['cutensor'] for g in groups.values())
    total_n = sum(g['n'] for g in groups.values())
    su = total_t / total_c if total_c > 0 else 0
    print("-" * 70)
    print(f"{'TOTAL':<12} {'':<12} {total_n:>5} {total_c:>10.2f} {total_t:>12.2f} {su:>7.2f}x")
    print()


# ---------------------------------------------------------------------------
# T2: Tier count table — block counts per symmetry
# ---------------------------------------------------------------------------

def t2_tier_counts(rows: List[PipelineRow]):
    """Aggregate large/medium/small block counts per symmetry."""
    groups: Dict[str, dict] = defaultdict(lambda: {'large': 0, 'medium': 0, 'small': 0})
    for r in rows:
        groups[r.symmetry]['large'] += r.large_problems
        groups[r.symmetry]['medium'] += r.medium_descs
        groups[r.symmetry]['small'] += r.small_descs

    print("=" * 70)
    print("T2: Tier Count Table (block counts per symmetry)")
    print("=" * 70)
    print(f"{'Symmetry':<12} {'Large':>8} {'Medium':>8} {'Small':>8} {'Total':>8} {'L%':>6} {'M%':>6} {'S%':>6}")
    print("-" * 70)

    for sym in sorted(groups):
        g = groups[sym]
        total = g['large'] + g['medium'] + g['small']
        if total == 0:
            continue
        lp = g['large'] / total * 100
        mp = g['medium'] / total * 100
        sp = g['small'] / total * 100
        print(f"{sym:<12} {g['large']:>8} {g['medium']:>8} {g['small']:>8} {total:>8} {lp:>5.1f}% {mp:>5.1f}% {sp:>5.1f}%")
    print()


# ---------------------------------------------------------------------------
# T3: Tier FLOP summary — FLOP fraction per tier per symmetry
# ---------------------------------------------------------------------------

def t3_tier_flops(rows: List[PipelineRow]):
    """Aggregate FLOPs per tier per symmetry, highlight small+medium fraction."""
    groups: Dict[str, dict] = defaultdict(lambda: {'large': 0.0, 'medium': 0.0, 'small': 0.0,
                                                    'large_n': 0, 'medium_n': 0, 'small_n': 0})
    for r in rows:
        groups[r.symmetry]['large'] += r.large_flops_gflop
        groups[r.symmetry]['medium'] += r.medium_flops_gflop
        groups[r.symmetry]['small'] += r.small_flops_gflop
        groups[r.symmetry]['large_n'] += r.large_problems
        groups[r.symmetry]['medium_n'] += r.medium_descs
        groups[r.symmetry]['small_n'] += r.small_descs

    print("=" * 70)
    print("T3: Tier FLOP Summary (per symmetry)")
    print("=" * 70)
    print(f"{'Symmetry':<12} {'L GFLOP':>10} {'M GFLOP':>10} {'S GFLOP':>10} {'L%':>6} {'M%':>6} {'S%':>6}")
    print("-" * 70)

    total_l = total_m = total_s = 0.0
    total_ln = total_mn = total_sn = 0
    for sym in sorted(groups):
        g = groups[sym]
        total = g['large'] + g['medium'] + g['small']
        if total == 0:
            continue
        lp = g['large'] / total * 100
        mp = g['medium'] / total * 100
        sp = g['small'] / total * 100
        print(f"{sym:<12} {g['large']:>10.2f} {g['medium']:>10.2f} {g['small']:>10.2f} {lp:>5.1f}% {mp:>5.1f}% {sp:>5.1f}%")
        total_l += g['large']; total_m += g['medium']; total_s += g['small']
        total_ln += g['large_n']; total_mn += g['medium_n']; total_sn += g['small_n']

    grand = total_l + total_m + total_s
    if grand > 0:
        sm_block_frac = (total_mn + total_sn) / (total_ln + total_mn + total_sn) * 100
        sm_flop_frac = (total_m + total_s) / grand * 100
        print("-" * 70)
        print(f"Small+medium handle {sm_block_frac:.0f}% of blocks but {sm_flop_frac:.1f}% of FLOPs")
    print()


# ---------------------------------------------------------------------------
# T4: Resource-ready prepare times
# ---------------------------------------------------------------------------

def t4_setup_times(rows: List[PipelineRow]):
    """Compare host planning separately from device initialization."""
    print("=" * 128)
    print("T4: Matched Prepare Time (post-allocation; host planning and device zero shown separately)")
    print("=" * 128)
    print(f"{'Case':<45} {'T host':>8} {'CT host':>8} {'Host su':>8} "
          f"{'T zero':>8} {'CT zero':>8} {'T total':>9} {'CT total':>9}")
    print("-" * 128)

    valid = [r for r in rows
             if r.tides_prepare_status == 'ok'
             and r.cutensor_prepare_status == 'ok']
    for r in sorted(valid, key=lambda x: x.case_name):
        ratio = (r.cutensor_host_prepare_ms / r.tides_host_prepare_ms
                 if r.tides_host_prepare_ms > 0 else 0)
        name = _short_name(r)
        print(f"{name:<45} {r.tides_host_prepare_ms:>8.3f} "
              f"{r.cutensor_host_prepare_ms:>8.3f} {ratio:>7.2f}x "
              f"{r.tides_device_zero_ms:>8.3f} "
              f"{r.cutensor_device_zero_ms:>8.3f} "
              f"{r.tides_prepare_ms:>9.3f} "
              f"{r.cutensor_prepare_ms:>9.3f}")

    # Aggregate by layout
    print()
    agg: Dict[str, dict] = defaultdict(lambda: {
        't_host': 0.0, 'ct_host': 0.0,
        't_zero': 0.0, 'ct_zero': 0.0,
        't_total': 0.0, 'ct_total': 0.0, 'n': 0,
    })
    for r in valid:
        g = agg[r.data_layout]
        g['t_host'] += r.tides_host_prepare_ms
        g['ct_host'] += r.cutensor_host_prepare_ms
        g['t_zero'] += r.tides_device_zero_ms
        g['ct_zero'] += r.cutensor_device_zero_ms
        g['t_total'] += r.tides_prepare_ms
        g['ct_total'] += r.cutensor_prepare_ms
        g['n'] += 1
    for layout in sorted(agg):
        g = agg[layout]
        ratio = g['ct_host'] / g['t_host'] if g['t_host'] > 0 else 0
        print(f"  [{layout}] n={g['n']} host avg T/CT="
              f"{g['t_host']/g['n']:.3f}/{g['ct_host']/g['n']:.3f} ms "
              f"({ratio:.2f}x); zero avg T/CT="
              f"{g['t_zero']/g['n']:.3f}/{g['ct_zero']/g['n']:.3f} ms; "
              f"total avg T/CT="
              f"{g['t_total']/g['n']:.3f}/{g['ct_total']/g['n']:.3f} ms")
    if valid:
        t_host = sum(r.tides_host_prepare_ms for r in valid)
        ct_host = sum(r.cutensor_host_prepare_ms for r in valid)
        t_zero = sum(r.tides_device_zero_ms for r in valid)
        ct_zero = sum(r.cutensor_device_zero_ms for r in valid)
        t_total = sum(r.tides_prepare_ms for r in valid)
        ct_total = sum(r.cutensor_prepare_ms for r in valid)
        tides_wins = sum(r.tides_host_prepare_ms < r.cutensor_host_prepare_ms
                         for r in valid)
        ratio = ct_host / t_host if t_host > 0 else 0
        print(f"  [overall] n={len(valid)} host T/CT={t_host:.3f}/{ct_host:.3f} ms "
              f"({ratio:.2f}x, TIDES wins {tides_wins}/{len(valid)}); "
              f"zero T/CT={t_zero:.3f}/{ct_zero:.3f} ms; "
              f"total T/CT={t_total:.3f}/{ct_total:.3f} ms")

    excluded = len(rows) - len(valid)
    if excluded:
        status_pairs = Counter(
            (r.tides_prepare_status, r.cutensor_prepare_status)
            for r in rows if r not in valid)
        details = ', '.join(
            f"T={t}/CT={ct}: {n}"
            for (t, ct), n in sorted(status_pairs.items()))
        print(f"  [excluded] {excluded}/{len(rows)} cases ({details})")
    print()


# ---------------------------------------------------------------------------
# T5: Transfer time estimate
# ---------------------------------------------------------------------------

def t5_transfer_estimate(rows: List[PipelineRow]):
    """Estimate H2D transfer time and compare to setup overhead."""
    H100_PCIE_GBS = 64.0   # GB/s PCIe gen5 x16

    print("=" * 90)
    print("T5: H2D Transfer Time Estimate (PCIe 64 GB/s)")
    print("=" * 90)
    print(f"{'Case':<40} {'H2D MB':>8} {'H2D ms':>8} {'Est ms':>8} {'Init ms':>9} {'Init/H2D':>9}")
    print("-" * 90)

    for r in sorted(rows, key=lambda x: x.case_name):
        if r.h2d_mb == 0:
            continue
        est_ms = r.h2d_mb / H100_PCIE_GBS * 1000  # MB / (GB/s) = ms
        init_ratio = r.init_ms / r.h2d_ms if r.h2d_ms > 0 else 0
        name = _short_name(r)
        print(f"{name:<40} {r.h2d_mb:>8.1f} {r.h2d_ms:>8.2f} {est_ms:>8.2f} {r.init_ms:>9.2f} {init_ratio:>8.1f}x")
    print()


# ---------------------------------------------------------------------------
# T6: Streaming workspace savings
# ---------------------------------------------------------------------------

def t6_streaming_workspace(rows: List[PipelineRow]):
    """Per case: streaming batches, buf size, vs peak memory."""
    print("=" * 90)
    print("T6: Streaming Workspace Savings")
    print("=" * 90)
    print(f"{'Case':<40} {'Batches':>8} {'Buf MB':>8} {'Peak MB':>9} {'CT Peak':>9} {'Tile Configs'}")
    print("-" * 90)

    for r in sorted(rows, key=lambda x: x.case_name):
        if r.streaming_batches == 0:
            continue
        name = _short_name(r)
        print(f"{name:<40} {r.streaming_batches:>8} {r.streaming_buf_mb:>8} "
              f"{r.peak_memory_mb:>9.1f} {r.cutensor_peak_mb:>9.1f} {r.streaming_tile_configs}")

    # Summary: cases where streaming saves >2x
    multi_batch = [r for r in rows if r.streaming_batches > 1]
    if multi_batch:
        print(f"\n  {len(multi_batch)} / {len(rows)} cases use multi-batch streaming")
        max_batch = max(r.streaming_batches for r in multi_batch)
        print(f"  Max batches: {max_batch}")
    print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_name(r: PipelineRow) -> str:
    """Shorten case name for display."""
    name = r.case_name
    if "_symm_" in name:
        parts = name.split("_symm_")
        tag = "can" if "canonical" in name else ("int" if "internal" in name else "mat")
        name = f"[{tag}] {parts[1]}"
    return name


TABLES = {
    'T1': t1_iteration_speedup,
    'T2': t2_tier_counts,
    'T3': t3_tier_flops,
    'T4': t4_setup_times,
    'T5': t5_transfer_estimate,
    'T6': t6_streaming_workspace,
}


def main():
    parser = argparse.ArgumentParser(description='Analyze pipeline benchmark results for paper.')
    parser.add_argument('input', help='Raw bench_pipeline_implementations output file')
    parser.add_argument('--only', nargs='+', metavar='T', choices=TABLES.keys(),
                        help='Only compute specified tables (e.g. T1 T3)')
    parser.add_argument('--layout', nargs='+', metavar='L',
                        help='Filter to specific layouts (canonical, int_perm, mat_perm)')
    args = parser.parse_args()

    layouts = set(args.layout) if args.layout else None
    rows = load_rows(args.input, layouts)

    if not rows:
        print("No pipeline rows parsed.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(rows)} cases\n", file=sys.stderr)

    tables = args.only or sorted(TABLES.keys())
    for t in tables:
        TABLES[t](rows)


if __name__ == '__main__':
    main()
