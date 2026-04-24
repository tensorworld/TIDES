#!/usr/bin/env python3
"""
Parse raw benchmark output into CSV.

Handles three benchmark formats:
  - bench_pipeline_implementations: per-case CUTLASS vs cuTENSOR timing
  - bench_transpose_cutensor: single/batched/mixed transpose comparison
  - bench_synthetic (fuzz_pipeline --scale): sweep-based scaling benchmarks

Usage:
    python scripts/parse_results.py output.txt                   # print CSV to stdout
    python scripts/parse_results.py output.txt -o results.csv    # write to file
    python scripts/parse_results.py output.txt --format table    # human-readable table
"""

import argparse
import csv
import io
import re
import sys
from dataclasses import dataclass, fields
from typing import List, Optional


# ---------------------------------------------------------------------------
# Pipeline benchmark parser
# ---------------------------------------------------------------------------

def extract_layout_info(case_name: str):
    """Extract data_layout, symmetry, size from case directory name."""
    if "canonical" in case_name:                  layout = "canonical"
    elif "block_permuted_matrices" in case_name:  layout = "mat_perm"
    elif "block_permuted_internal" in case_name:  layout = "int_perm"
    else: layout = ""
    m = re.search(r'_symm_(\w+?)_(E\d|S\d)', case_name)
    symmetry = m.group(1) if m else ""
    size = m.group(2) if m else ""
    return layout, symmetry, size


@dataclass
class PipelineRow:
    case_name: str
    einsum: str
    a_blocks: int
    b_blocks: int
    layout: str
    cutlass_ms: float
    cutensor_ms: float
    speedup: float
    transpose_ms: float
    contract_ms: float
    accumulate_ms: float
    large_problems: int
    medium_descs: int
    small_descs: int
    total_flops_gflop: float
    peak_memory_mb: float
    # EXP-5: H2D transfer
    h2d_ms: float = 0.0
    h2d_mb: float = 0.0
    # EXP-2: setup fairness
    cutlass_setup_ms: float = 0.0
    # EXP-4: per-tier FLOPs and output elements
    large_flops_gflop: float = 0.0
    medium_flops_gflop: float = 0.0
    small_flops_gflop: float = 0.0
    large_output_elems: int = 0
    medium_output_elems: int = 0
    small_output_elems: int = 0
    # EXP-3: layout/symmetry decomposition
    data_layout: str = ""
    symmetry: str = ""
    size: str = ""
    # EXP-5: setup sub-components
    preproc_ms: float = 0.0
    accum_map_ms: float = 0.0
    bs_setup_ms: float = 0.0
    init_ms: float = 0.0
    # cuTENSOR detail
    cutensor_setup_ms: float = 0.0
    cutensor_peak_mb: float = 0.0
    # E2E detail
    gflops_per_s: float = 0.0
    # Streaming
    streaming_batches: int = 0
    streaming_buf_mb: int = 0
    streaming_tile_configs: str = ""


def parse_pipeline(text: str) -> List[PipelineRow]:
    """Parse bench_pipeline_implementations output."""
    rows: List[PipelineRow] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # Case header: "case_name  einsum=...  A_blk=N  B_blk=M"
        m = re.match(
            r'^(\S+)\s+einsum=(\S+)\s+A_blk=(\d+)\s+B_blk=(\d+)', line
        )
        if not m:
            i += 1
            continue

        case_name = m.group(1)
        einsum = m.group(2)
        a_blocks = int(m.group(3))
        b_blocks = int(m.group(4))

        # Collect subsequent indented lines for this case
        layout = ""
        cutlass_ms = 0.0
        cutensor_ms = 0.0
        transpose_ms = 0.0
        contract_ms = 0.0
        accumulate_ms = 0.0
        large_problems = 0
        medium_descs = 0
        small_descs = 0
        total_flops = 0.0
        peak_mb = 0.0
        h2d_ms = 0.0
        h2d_mb = 0.0
        cutlass_setup_ms = 0.0
        large_flops = 0.0
        medium_flops = 0.0
        small_flops = 0.0
        large_output = 0
        medium_output = 0
        small_output = 0
        preproc_ms = 0.0
        accum_map_ms = 0.0
        bs_setup_ms = 0.0
        init_ms = 0.0
        cutensor_setup_ms = 0.0
        cutensor_peak_mb = 0.0
        gflops_per_s = 0.0
        streaming_batches = 0
        streaming_buf_mb = 0
        streaming_tile_configs = ""

        i += 1
        while i < len(lines) and (lines[i].startswith('  ') or lines[i] == ''):
            ln = lines[i].strip()

            # layout=TN ...
            m2 = re.match(r'layout=(\S+)', ln)
            if m2:
                layout = m2.group(1)

            # cutlass=  N.NNN ms  bs_cutensor=  N.NNN ms
            m2 = re.match(
                r'cutlass=\s*([\d.]+)\s*ms\s+bs_cutensor=\s*([\d.]+)\s*ms', ln
            )
            if m2:
                cutlass_ms = float(m2.group(1))
                cutensor_ms = float(m2.group(2))

            # cutlass= with cutensor error
            m2 = re.match(r'cutlass=\s*([\d.]+)\s*ms\s+bs_cutensor=\s*ERROR', ln)
            if m2:
                cutlass_ms = float(m2.group(1))
                cutensor_ms = -1.0

            # [cutlass] transpose=0.11 contract=2.16 accumulate=0.01 total=2.31 ms
            m2 = re.match(
                r'\[cutlass\]\s+(?:transpose=([\d.]+)\s+)?'
                r'contract=([\d.]+)\s+accumulate=([\d.]+)\s+total=[\d.]+\s*ms',
                ln,
            )
            if m2:
                transpose_ms = float(m2.group(1)) if m2.group(1) else 0.0
                contract_ms = float(m2.group(2))
                accumulate_ms = float(m2.group(3))

            # [large] problems=145  flops=22.15 GFLOP  output=12345 elems
            m2 = re.search(r'\[large\]\s+problems=(\d+)\s+flops=([\d.]+)\s+GFLOP\s+output=(\d+)', ln)
            if m2:
                large_problems = int(m2.group(1))
                large_flops = float(m2.group(2))
                large_output = int(m2.group(3))
            elif re.search(r'\[large\]\s+problems=(\d+)', ln):
                large_problems = int(re.search(r'\[large\]\s+problems=(\d+)', ln).group(1))

            # [medium] descs=3  flops=0.50 GFLOP  output=678 elems
            m2 = re.search(r'\[medium\]\s+descs=(\d+)\s+flops=([\d.]+)\s+GFLOP\s+output=(\d+)', ln)
            if m2:
                medium_descs = int(m2.group(1))
                medium_flops = float(m2.group(2))
                medium_output = int(m2.group(3))
            elif re.search(r'\[medium\]\s+descs=(\d+)', ln):
                medium_descs = int(re.search(r'\[medium\]\s+descs=(\d+)', ln).group(1))

            # [small] descs=13 ... flops=0.02 GFLOP  output=89 elems
            m2 = re.search(r'\[small\]\s+descs=(\d+).*flops=([\d.]+)\s+GFLOP\s+output=(\d+)', ln)
            if m2:
                small_descs = int(m2.group(1))
                small_flops = float(m2.group(2))
                small_output = int(m2.group(3))
            elif re.search(r'\[small\]\s+descs=(\d+)', ln):
                small_descs = int(re.search(r'\[small\]\s+descs=(\d+)', ln).group(1))

            # [setup] cutlass_setup=X.XX preproc=X.XX accum_map=X.XX bs_setup=X.XX h2d=X.XX h2d_MB=X.X init=X.XX ms
            m2 = re.search(r'cutlass_setup=([\d.]+)', ln)
            if m2:
                cutlass_setup_ms = float(m2.group(1))
            m2 = re.search(r'preproc=([\d.]+)', ln)
            if m2:
                preproc_ms = float(m2.group(1))
            m2 = re.search(r'accum_map=([\d.]+)', ln)
            if m2:
                accum_map_ms = float(m2.group(1))
            m2 = re.search(r'bs_setup=([\d.]+)', ln)
            if m2:
                bs_setup_ms = float(m2.group(1))
            m2 = re.search(r'\bh2d=([\d.]+)', ln)
            if m2:
                h2d_ms = float(m2.group(1))
            m2 = re.search(r'h2d_MB=([\d.]+)', ln)
            if m2:
                h2d_mb = float(m2.group(1))
            m2 = re.search(r'init=([\d.]+)', ln)
            if m2:
                init_ms = float(m2.group(1))

            # [e2e] flops=22.15 GFLOP  GFLOP/s=1243.4
            m2 = re.search(r'\[e2e\]\s+flops=([\d.]+)\s+GFLOP', ln)
            if m2:
                total_flops = float(m2.group(1))
            m2 = re.search(r'GFLOP/s=([\d.]+)', ln)
            if m2:
                gflops_per_s = float(m2.group(1))

            # [bs_cutensor] setup=1.64 kernel=0.27 ms  peak=41.4 MB
            m2 = re.match(r'\[bs_cutensor\]\s+setup=([\d.]+)\s+kernel=[\d.]+\s*ms\s+peak=([\d.]+)', ln)
            if m2:
                cutensor_setup_ms = float(m2.group(1))
                cutensor_peak_mb = float(m2.group(2))

            # [streaming] 2 batches  buf=7 MB x2  precomp=0 KB  Square=1925p/1b/32x32  SkinnyTall=...
            m2 = re.match(r'\[streaming\]\s+(\d+)\s+batch', ln)
            if m2:
                streaming_batches = int(m2.group(1))
                m3 = re.search(r'buf=(\d+)\s*MB', ln)
                if m3:
                    streaming_buf_mb = int(m3.group(1))
                # Tile configs: everything after precomp=... KB
                m3 = re.search(r'precomp=\S+\s+KB\s+(.*)', ln)
                if m3:
                    streaming_tile_configs = m3.group(1).strip()

            # [memory] peak=176.9 MB
            m2 = re.search(r'\[memory\]\s+peak=([\d.]+)\s*MB', ln)
            if m2:
                peak_mb = float(m2.group(1))

            i += 1

        speedup = cutlass_ms / cutensor_ms if cutensor_ms > 0 else 0.0
        data_layout, symmetry, size = extract_layout_info(case_name)

        rows.append(PipelineRow(
            case_name=case_name,
            einsum=einsum,
            a_blocks=a_blocks,
            b_blocks=b_blocks,
            layout=layout,
            cutlass_ms=cutlass_ms,
            cutensor_ms=cutensor_ms,
            speedup=round(speedup, 3),
            transpose_ms=transpose_ms,
            contract_ms=contract_ms,
            accumulate_ms=accumulate_ms,
            large_problems=large_problems,
            medium_descs=medium_descs,
            small_descs=small_descs,
            total_flops_gflop=total_flops,
            peak_memory_mb=peak_mb,
            h2d_ms=h2d_ms,
            h2d_mb=h2d_mb,
            cutlass_setup_ms=cutlass_setup_ms,
            large_flops_gflop=large_flops,
            medium_flops_gflop=medium_flops,
            small_flops_gflop=small_flops,
            large_output_elems=large_output,
            medium_output_elems=medium_output,
            small_output_elems=small_output,
            data_layout=data_layout,
            symmetry=symmetry,
            size=size,
            preproc_ms=preproc_ms,
            accum_map_ms=accum_map_ms,
            bs_setup_ms=bs_setup_ms,
            init_ms=init_ms,
            cutensor_setup_ms=cutensor_setup_ms,
            cutensor_peak_mb=cutensor_peak_mb,
            gflops_per_s=gflops_per_s,
            streaming_batches=streaming_batches,
            streaming_buf_mb=streaming_buf_mb,
            streaming_tile_configs=streaming_tile_configs,
        ))

    return rows


# ---------------------------------------------------------------------------
# Transpose benchmark parser
# ---------------------------------------------------------------------------

@dataclass
class TransposeSingleRow:
    section: str  # "single"
    perm: str
    shape: str
    cutensor_ms: float
    hypertile_ms: float
    speedup: float
    ht_gbps: float
    ct_gbps: float


@dataclass
class TransposeBatchedRow:
    section: str  # "batched"
    perm: str
    block_shape: str
    num_blocks: int
    cutensor_ms: float
    hypertile_ms: float
    speedup: float
    ht_gbps: float
    ct_gbps: float


@dataclass
class TransposeMixedRow:
    section: str  # "mixed"
    perm: str
    avg_vol: int
    num_blocks: int
    rank: int
    cutensor_ms: float
    hypertile_ms: float
    speedup: float


@dataclass
class TransposeBatchSweepRow:
    section: str  # "batch_sweep"
    mode: str     # "fixed" or "mixed"
    num_blocks: int
    block_vol: int
    cutensor_ms: float
    hypertile_ms: float
    speedup: float


@dataclass
class TransposeGroupedRow:
    section: str  # "grouped"
    case_name: str
    num_shapes: int
    num_blocks: int
    total_vol: str
    cutensor_ms: float
    hypertile_ms: float
    speedup: float
    ht_gbps: float
    ct_gbps: float


@dataclass
class TransposeExp1bARow:
    section: str  # "exp1b_a"
    shape: str
    perm: str
    wr: int
    num_blocks: int
    total_vol: str
    cutensor_ms: float
    hypertile_ms: float
    speedup: float
    ht_gbps: float
    ct_gbps: float


@dataclass
class TransposeExp1bBRow:
    section: str  # "exp1b_b"
    case_name: str
    operand: str
    perm: str
    num_shapes: int
    num_blocks: int
    total_vol: str
    cutensor_ms: float
    hypertile_ms: float
    speedup: float
    ht_gbps: float
    ct_gbps: float


def parse_transpose(text: str):
    """Parse bench_transpose_cutensor output. Returns list of row dicts."""
    rows = []
    lines = text.splitlines()
    section = None

    for line in lines:
        # Detect section headers
        if 'EXP-1b(a)' in line:
            section = 'exp1b_a'
            continue
        elif 'EXP-1b(b)' in line:
            section = 'exp1b_b'
            continue
        elif 'Grouped Transpose' in line:
            section = 'grouped'
            continue
        elif 'Single Tensor Transpose' in line:
            section = 'single'
            continue
        elif 'Batch Sweep Transpose' in line:
            section = 'batch_sweep'
            continue
        elif 'Batched Transpose' in line:
            section = 'batched'
            continue
        elif 'Mixed Block Sizes' in line:
            section = 'mixed'
            continue

        # Skip headers/separators
        if line.startswith('----') or line.startswith('Perm') or not line.strip():
            continue
        if line.startswith('Device:') or line.startswith('Settings:'):
            continue
        # Skip comment lines from run_bench wrapper
        if line.startswith('#'):
            continue

        if section == 'single':
            # 1,0          2048x2048                  0.074ms     0.072ms      1.02x   467.1   455.7
            # or: skip / FAIL lines
            m = re.match(
                r'(\S+)\s+(\S+)\s+'
                r'([\d.]+)ms\s+([\d.]+)ms\s+([\d.]+)x\s+([\d.]+)\s+([\d.]+)',
                line,
            )
            if m:
                rows.append(vars(TransposeSingleRow(
                    section='single',
                    perm=m.group(1),
                    shape=m.group(2),
                    cutensor_ms=float(m.group(3)),
                    hypertile_ms=float(m.group(4)),
                    speedup=float(m.group(5)),
                    ht_gbps=float(m.group(6)),
                    ct_gbps=float(m.group(7)),
                )))
                continue

            # cuTENSOR with hypertile skip
            m = re.match(
                r'(\S+)\s+(\S+)\s+([\d.]+)ms\s+skip', line,
            )
            if m:
                rows.append(vars(TransposeSingleRow(
                    section='single',
                    perm=m.group(1),
                    shape=m.group(2),
                    cutensor_ms=float(m.group(3)),
                    hypertile_ms=-1.0,
                    speedup=0.0,
                    ht_gbps=0.0,
                    ct_gbps=0.0,
                )))

        elif section == 'batched':
            # 1,2,0        6x6x7                       100     0.379ms     0.024ms     16.09x     8.6     0.5
            m = re.match(
                r'(\S+)\s+(\S+)\s+(\d+)\s+'
                r'([\d.]+)ms\s+([\d.]+)ms\s+([\d.]+)x\s+([\d.]+)\s+([\d.]+)',
                line,
            )
            if m:
                rows.append(vars(TransposeBatchedRow(
                    section='batched',
                    perm=m.group(1),
                    block_shape=m.group(2),
                    num_blocks=int(m.group(3)),
                    cutensor_ms=float(m.group(4)),
                    hypertile_ms=float(m.group(5)),
                    speedup=float(m.group(6)),
                    ht_gbps=float(m.group(7)),
                    ct_gbps=float(m.group(8)),
                )))
                continue

            # OOM line
            m = re.match(r'(\S+)\s+(\S+)\s+(\d+)\s+OOM', line)
            if m:
                rows.append(vars(TransposeBatchedRow(
                    section='batched',
                    perm=m.group(1),
                    block_shape=m.group(2),
                    num_blocks=int(m.group(3)),
                    cutensor_ms=-1.0,
                    hypertile_ms=-1.0,
                    speedup=0.0,
                    ht_gbps=0.0,
                    ct_gbps=0.0,
                )))

        elif section == 'mixed':
            # reverse   20529       10      3     0.040ms     0.035ms      1.15x
            m = re.match(
                r'(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+'
                r'([\d.]+)ms\s+([\d.]+)ms\s+([\d.]+)x',
                line,
            )
            if m:
                rows.append(vars(TransposeMixedRow(
                    section='mixed',
                    perm=m.group(1),
                    avg_vol=int(m.group(2)),
                    num_blocks=int(m.group(3)),
                    rank=int(m.group(4)),
                    cutensor_ms=float(m.group(5)),
                    hypertile_ms=float(m.group(6)),
                    speedup=float(m.group(7)),
                )))

        elif section == 'batch_sweep':
            # fixed         5   51380224     1.234ms     0.567ms      2.18x
            m = re.match(
                r'(\S+)\s+(\d+)\s+(\d+)\s+'
                r'([\d.]+)ms\s+([\d.]+)ms\s+([\d.]+)x',
                line,
            )
            if m:
                rows.append(vars(TransposeBatchSweepRow(
                    section='batch_sweep',
                    mode=m.group(1),
                    num_blocks=int(m.group(2)),
                    block_vol=int(m.group(3)),
                    cutensor_ms=float(m.group(4)),
                    hypertile_ms=float(m.group(5)),
                    speedup=float(m.group(6)),
                )))
                continue

            # OOM line
            m = re.match(r'(\S+)\s+(\d+)\s+(\d+)\s+OOM', line)
            if m:
                rows.append(vars(TransposeBatchSweepRow(
                    section='batch_sweep',
                    mode=m.group(1),
                    num_blocks=int(m.group(2)),
                    block_vol=int(m.group(3)),
                    cutensor_ms=-1.0,
                    hypertile_ms=-1.0,
                    speedup=0.0,
                )))

        elif section == 'grouped':
            # case_name   3   145   12.3M   2.45ms   0.31ms   7.9x   410.2   52.1
            m = re.match(
                r'(\S+)\s+(\d+)\s+(\d+)\s+(\S+)\s+'
                r'([\d.]+)ms\s+([\d.]+)ms\s+([\d.]+)x\s+([\d.]+)\s+([\d.]+)',
                line,
            )
            if m:
                rows.append(vars(TransposeGroupedRow(
                    section='grouped',
                    case_name=m.group(1),
                    num_shapes=int(m.group(2)),
                    num_blocks=int(m.group(3)),
                    total_vol=m.group(4),
                    cutensor_ms=float(m.group(5)),
                    hypertile_ms=float(m.group(6)),
                    speedup=float(m.group(7)),
                    ht_gbps=float(m.group(8)),
                    ct_gbps=float(m.group(9)),
                )))

        elif section == 'exp1b_a':
            # 373x6x200x2  1,0,2,3   3    146   130.7M   2.507ms   1.874ms   1.34x   558.0   417.1
            m = re.match(
                r'(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\S+)\s+'
                r'([\d.]+)ms\s+([\d.]+)ms\s+([\d.]+)x\s+([\d.]+)\s+([\d.]+)',
                line,
            )
            if m:
                rows.append(vars(TransposeExp1bARow(
                    section='exp1b_a',
                    shape=m.group(1),
                    perm=m.group(2),
                    wr=int(m.group(3)),
                    num_blocks=int(m.group(4)),
                    total_vol=m.group(5),
                    cutensor_ms=float(m.group(6)),
                    hypertile_ms=float(m.group(7)),
                    speedup=float(m.group(8)),
                    ht_gbps=float(m.group(9)),
                    ct_gbps=float(m.group(10)),
                )))

        elif section == 'exp1b_b':
            # case_name  A  1,0,2   510  1717  836.3K  6.131ms  0.053ms  115.14x  125.6  1.1
            m = re.match(
                r'(\S+)\s+([AB])\s+(\S+)\s+(\d+)\s+(\d+)\s+(\S+)\s+'
                r'([\d.]+)ms\s+([\d.]+)ms\s+([\d.]+)x\s+([\d.]+)\s+([\d.]+)',
                line,
            )
            if m:
                rows.append(vars(TransposeExp1bBRow(
                    section='exp1b_b',
                    case_name=m.group(1),
                    operand=m.group(2),
                    perm=m.group(3),
                    num_shapes=int(m.group(4)),
                    num_blocks=int(m.group(5)),
                    total_vol=m.group(6),
                    cutensor_ms=float(m.group(7)),
                    hypertile_ms=float(m.group(8)),
                    speedup=float(m.group(9)),
                    ht_gbps=float(m.group(10)),
                    ct_gbps=float(m.group(11)),
                )))

    return rows


# ---------------------------------------------------------------------------
# Synthetic benchmark parser (fuzz_pipeline --scale)
# ---------------------------------------------------------------------------

def parse_synthetic(text: str):
    """Parse bench_synthetic / fuzz_pipeline --scale output."""
    rows = []
    lines = text.splitlines()
    config = ""
    sweep = ""
    tier = ""

    for line in lines:
        # Config header: "# === fig1: L-tensor 2fA+2fB+2c [uniform] ==="
        m = re.match(r'^#\s*===\s*(.+?)\s*===\s*$', line)
        if m:
            config = m.group(1)
            continue

        # Sweep header: "=== Sweep: section_size (tier=all, ...) ==="
        m = re.match(r'^=== Sweep:\s*(\S+)\s*\(tier=(\S+)', line)
        if m:
            sweep = m.group(1)
            tier = m.group(2).rstrip(',')
            continue

        # Data row (new format with M/N/K):
        #   4.0       16       16       16     17     17     40  0/40/0  0.21  0.02  0.88  0.06  2.5x  4.0x  PASS
        m = re.match(
            r'\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)'
            r'\s+(\d+)\s+(\d+)\s+(\d+)'
            r'\s+(\d+)/(\d+)/(\d+)'
            r'\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
            r'\s+([\d.]+)x\s+([\d.]+)x'
            r'\s+(\S+)',
            line,
        )
        if m:
            rows.append({
                'config': config,
                'sweep': sweep,
                'tier': tier,
                'param': float(m.group(1)),
                'M': int(m.group(2)),
                'N': int(m.group(3)),
                'K': int(m.group(4)),
                'a_blk': int(m.group(5)),
                'b_blk': int(m.group(6)),
                'pairs': int(m.group(7)),
                'large': int(m.group(8)),
                'medium': int(m.group(9)),
                'small': int(m.group(10)),
                'setup_ms': float(m.group(11)),
                'exec_ms': float(m.group(12)),
                'ct_setup_ms': float(m.group(13)),
                'ct_kern_ms': float(m.group(14)),
                'exec_speedup': float(m.group(15)),
                'total_speedup': float(m.group(16)),
                'status': m.group(17),
            })
            continue

        # Data row (old format without M/N/K):
        #   12.0    260    260   1901     1901/0/0      25.04       1.12       0.28       0.90     0.8x     0.0x   PASS
        m = re.match(
            r'\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)'
            r'\s+(\d+)/(\d+)/(\d+)'
            r'\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
            r'\s+([\d.]+)x\s+([\d.]+)x'
            r'\s+(\S+)',
            line,
        )
        if m:
            rows.append({
                'config': config,
                'sweep': sweep,
                'tier': tier,
                'param': float(m.group(1)),
                'M': 0, 'N': 0, 'K': 0,
                'a_blk': int(m.group(2)),
                'b_blk': int(m.group(3)),
                'pairs': int(m.group(4)),
                'large': int(m.group(5)),
                'medium': int(m.group(6)),
                'small': int(m.group(7)),
                'setup_ms': float(m.group(8)),
                'exec_ms': float(m.group(9)),
                'ct_setup_ms': float(m.group(10)),
                'ct_kern_ms': float(m.group(11)),
                'exec_speedup': float(m.group(12)),
                'total_speedup': float(m.group(13)),
                'status': m.group(14),
            })
            continue

        # SKIP row:  "  20.0   SKIP (pairs 2559230 > cap 200000)"
        m = re.match(r'\s+([\d.]+)\s+SKIP\s*\((.+)\)', line)
        if m:
            rows.append({
                'config': config,
                'sweep': sweep,
                'tier': tier,
                'param': float(m.group(1)),
                'M': 0, 'N': 0, 'K': 0,
                'a_blk': 0, 'b_blk': 0, 'pairs': 0,
                'large': 0, 'medium': 0, 'small': 0,
                'setup_ms': 0, 'exec_ms': 0,
                'ct_setup_ms': 0, 'ct_kern_ms': 0,
                'exec_speedup': 0, 'total_speedup': 0,
                'status': 'SKIP',
            })
            continue

        # ERROR row: "  38.0   ERROR (malloc final_output: out of memory)"
        m = re.match(r'\s+([\d.]+)\s+ERROR\s*\((.+)\)', line)
        if m:
            rows.append({
                'config': config,
                'sweep': sweep,
                'tier': tier,
                'param': float(m.group(1)),
                'M': 0, 'N': 0, 'K': 0,
                'a_blk': 0, 'b_blk': 0, 'pairs': 0,
                'large': 0, 'medium': 0, 'small': 0,
                'setup_ms': 0, 'exec_ms': 0,
                'ct_setup_ms': 0, 'ct_kern_ms': 0,
                'exec_speedup': 0, 'total_speedup': 0,
                'status': 'ERROR',
            })

    return rows


# ---------------------------------------------------------------------------
# Detect format and dispatch
# ---------------------------------------------------------------------------

def detect_format(text: str) -> str:
    """Detect which benchmark produced this output."""
    if 'Pipeline Implementation Benchmark' in text:
        return 'pipeline'
    if ('cuTENSOR vs Hypertile' in text or 'Batched Transpose' in text
            or 'Batch Sweep Transpose' in text or 'Grouped Transpose' in text):
        return 'transpose'
    if '=== Sweep:' in text:
        return 'synthetic'
    return 'unknown'


def to_csv(rows, out):
    """Write list of row dicts to CSV."""
    if not rows:
        return
    # For pipeline rows (dataclass instances), convert to dicts
    if hasattr(rows[0], '__dataclass_fields__'):
        rows = [vars(r) for r in rows]
    # Collect all keys across all rows (mixed transpose has different row types)
    all_keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    writer = csv.DictWriter(out, fieldnames=all_keys, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)


def to_table(rows):
    """Render rows as an aligned text table."""
    if not rows:
        return ""
    if hasattr(rows[0], '__dataclass_fields__'):
        rows = [vars(r) for r in rows]
    all_keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    col_widths = {k: len(k) for k in all_keys}
    for r in rows:
        for k in all_keys:
            val = str(r.get(k, ''))
            col_widths[k] = max(col_widths[k], len(val))
    header = '  '.join(k.ljust(col_widths[k]) for k in all_keys)
    sep = '  '.join('-' * col_widths[k] for k in all_keys)
    lines = [header, sep]
    for r in rows:
        lines.append('  '.join(
            str(r.get(k, '')).ljust(col_widths[k]) for k in all_keys
        ))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_select_list(select_arg: str) -> List[str]:
    """Parse --select: comma-separated names or @filename."""
    if select_arg.startswith('@'):
        path = select_arg[1:]
        with open(path) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [s.strip() for s in select_arg.split(',') if s.strip()]


def _filter_pipeline_rows(rows: List[PipelineRow], names: List[str]) -> List[PipelineRow]:
    """Filter pipeline rows by case_name substring match."""
    filtered = []
    for r in rows:
        if any(n in r.case_name for n in names):
            filtered.append(r)
    return filtered


def _tier_summary(rows: List[PipelineRow]):
    """Print a compact tier diversity summary to help cherry-pick cases."""
    print("\n=== Tier Diversity Summary (for cherry-picking) ===\n", file=sys.stderr)
    print(f"{'Case':<50} {'L':>5} {'M':>5} {'S':>5} {'L%':>5} {'M%':>5} {'S%':>5} "
          f"{'Trans':>5} {'su':>6}", file=sys.stderr)
    print("-" * 98, file=sys.stderr)
    for r in rows:
        total = r.large_flops_gflop + r.medium_flops_gflop + r.small_flops_gflop
        lp = (r.large_flops_gflop / total * 100) if total > 0 else 0
        mp = (r.medium_flops_gflop / total * 100) if total > 0 else 0
        sp = (r.small_flops_gflop / total * 100) if total > 0 else 0
        has_trans = 'Y' if r.transpose_ms > 0.01 else 'N'
        # Shorten case name
        name = r.case_name
        if "_symm_" in name:
            parts = name.split("_symm_")
            cat = "can" if "canonical" in name else ("int" if "internal" in name else "mat")
            name = f"[{cat}] {parts[1]}"
        print(f"{name:<50} {r.large_problems:>5} {r.medium_descs:>5} {r.small_descs:>5} "
              f"{lp:>4.0f}% {mp:>4.0f}% {sp:>4.0f}% {has_trans:>5} {r.speedup:>5.2f}x",
              file=sys.stderr)
    print(file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Parse benchmark output to CSV or table.'
    )
    parser.add_argument('input', help='Raw benchmark output file')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument(
        '--format', choices=['csv', 'table'], default='csv',
        help='Output format (default: csv)',
    )
    # Convenience alias: --csv is the same as --format csv (for common.sh)
    parser.add_argument('--csv', action='store_true', help='Alias for --format csv')
    parser.add_argument(
        '--select', metavar='NAMES',
        help='Filter pipeline rows: comma-separated substrings or @filename',
    )
    parser.add_argument(
        '--tier-summary', action='store_true',
        help='Print tier diversity summary to stderr (for cherry-picking)',
    )
    args = parser.parse_args()

    with open(args.input) as f:
        text = f.read()

    fmt = detect_format(text)
    if fmt == 'pipeline':
        rows = parse_pipeline(text)
    elif fmt == 'transpose':
        rows = parse_transpose(text)
    elif fmt == 'synthetic':
        rows = parse_synthetic(text)
    else:
        print(f"Cannot detect benchmark format in {args.input}", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("No data rows parsed.", file=sys.stderr)
        sys.exit(1)

    # Pipeline-specific: tier summary and selection
    if fmt == 'pipeline' and args.tier_summary:
        _tier_summary(rows)

    if fmt == 'pipeline' and args.select:
        names = _load_select_list(args.select)
        rows = _filter_pipeline_rows(rows, names)
        if not rows:
            print(f"No rows matched --select filter.", file=sys.stderr)
            sys.exit(1)
        print(f"Selected {len(rows)} rows.", file=sys.stderr)

    out_format = 'csv' if args.csv else args.format

    if args.output:
        with open(args.output, 'w', newline='') as f:
            if out_format == 'csv':
                to_csv(rows, f)
            else:
                f.write(to_table(rows))
                f.write('\n')
        print(f"Wrote {len(rows)} rows to {args.output}", file=sys.stderr)
    else:
        if out_format == 'csv':
            to_csv(rows, sys.stdout)
        else:
            print(to_table(rows))


if __name__ == '__main__':
    main()
