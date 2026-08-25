#pragma once

#include "pipeline_descs.hpp"
#include "gpu_sparse_input.hpp"
#include "gpu_accumulator.cuh"
#include "ob_grouping.hpp"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>

namespace bsptc {

struct TransposeBatch {
    std::vector<int> problem_indices;   // into large.problems
    int64_t transpose_volume;           // total A+B elements to transpose in this batch

    // Precomputed pack layout: unique block indices and their offsets in d_buf.
    // pack_offset_A/B are prefix sums of block volumes; A packed first, then B.
    // local_A[i] / local_B[i] = index into unique_blocks_A/B for problem_indices[i].
    std::vector<int>     unique_blocks_A, unique_blocks_B;
    std::vector<int64_t> pack_offset_A, pack_offset_B;
    std::vector<int>     local_A, local_B;
};

struct BatchPartitionResult {
    std::vector<TransposeBatch> batches;
    int64_t max_batch_transpose_volume = 0;
    int64_t max_batch_expanded_problems = 0;

    // Output-block CSR over large.problems, built once by
    // partition_large_smart's K-grouping and reused by
    // compute_pair_separation (which otherwise rebuilds + re-sorts it).
    OBGroupResult ob_groups;
    bool has_ob_groups = false;

    // Per-problem (indexed by problem index in large.problems) flag:
    // 1 if all contributing pairs for the output block land in different batches
    // (separated), meaning the pair can write directly to d_output with beta=1.
    std::vector<uint8_t> pair_separated;
};

// Compute per-pair separation flags for an existing partition result.
// A pair is separated when its output block has all contributing pairs
// in different batches, so no two write to the same output region within
// a single GEMM call.
template <typename T>
void compute_pair_separation(
    BatchPartitionResult& result,
    const LargeBlockWork<T>& large,
    const AccumulationMap& accum_map)
{
    const int n = static_cast<int>(large.problems.size());
    result.pair_separated.assign(n, 0);

    if (result.batches.size() <= 1) return;

    // Map each problem to its batch index
    std::vector<int> prob_batch(n, -1);
    for (int bi = 0; bi < static_cast<int>(result.batches.size()); ++bi)
        for (int pi : result.batches[bi].problem_indices)
            prob_batch[pi] = bi;

    // Reuse the output-block CSR from partition_large_smart when present;
    // only the naive-partition path builds it here.
    OBGroupResult local;
    const OBGroupResult* groups = &result.ob_groups;
    if (!result.has_ob_groups) {
        std::vector<int64_t> ob_keys(n);
        for (int i = 0; i < n; ++i)
            ob_keys[i] = accum_map.h_pair_output_block_idx[large.problems[i].pair_idx];
        local = group_by_int64_key(ob_keys.data(), n);
        groups = &local;
    }

    // Generation-stamped batch marks (group id as generation) instead of a
    // per-group hash set.
    std::vector<int> batch_mark(result.batches.size(), -1);
    for (int g = 0; g < groups->num_groups; ++g) {
        if (groups->group_size(g) <= 1) continue;
        const int* begin = groups->group_begin(g);
        const int* end = groups->group_end(g);
        bool separated = true;
        for (const int* it = begin; it != end; ++it) {
            const int bi = prob_batch[*it];
            if (batch_mark[bi] == g) { separated = false; break; }
            batch_mark[bi] = g;
        }
        if (separated) {
            for (const int* it = begin; it != end; ++it)
                result.pair_separated[*it] = 1;
        }
    }
}

// Greedy capacity scan: walk problem_order, accumulate unique A+B volume and
// build per-batch pack layout (unique blocks, pack offsets, local indices).
// Flush batch when buffer cap exceeded. Shared by naive and smart partitioners.
template <typename T>
BatchPartitionResult greedy_batch_scan(
    const LargeBlockWork<T>& large,
    const GPUSparseInput<T>& A_gpu, const GPUSparseInput<T>& B_gpu,
    const int* problem_order, int num_problems,
    int64_t buffer_cap_elements)
{
    BatchPartitionResult result;

    if (num_problems == 0) return result;

    // Helper: add a problem to the current batch, tracking unique blocks and
    // pack offsets. Per-batch block dedup uses generation-stamped arrays
    // indexed by block id (batch id as generation) — one allocation for the
    // whole scan, no per-batch hash maps.
    TransposeBatch current;
    std::vector<int> gen_A(static_cast<size_t>(A_gpu.num_blocks), -1);
    std::vector<int> gen_B(static_cast<size_t>(B_gpu.num_blocks), -1);
    std::vector<int> slot_A(static_cast<size_t>(A_gpu.num_blocks));
    std::vector<int> slot_B(static_cast<size_t>(B_gpu.num_blocks));
    int batch_gen = 0;
    int64_t current_vol = 0, current_expanded = 0;

    auto add_problem = [&](int idx) {
        const auto& prob = large.problems[idx];
        current.problem_indices.push_back(idx);
        current_expanded += prob.batch_size;

        if (gen_A[prob.block_idx_A] != batch_gen) {
            gen_A[prob.block_idx_A] = batch_gen;
            slot_A[prob.block_idx_A] = static_cast<int>(current.unique_blocks_A.size());
            current.unique_blocks_A.push_back(prob.block_idx_A);
            current.pack_offset_A.push_back(current_vol);
            current_vol += A_gpu.block_volumes[prob.block_idx_A];
        }
        current.local_A.push_back(slot_A[prob.block_idx_A]);

        if (gen_B[prob.block_idx_B] != batch_gen) {
            gen_B[prob.block_idx_B] = batch_gen;
            slot_B[prob.block_idx_B] = static_cast<int>(current.unique_blocks_B.size());
            current.unique_blocks_B.push_back(prob.block_idx_B);
            current.pack_offset_B.push_back(current_vol);
            current_vol += B_gpu.block_volumes[prob.block_idx_B];
        }
        current.local_B.push_back(slot_B[prob.block_idx_B]);
    };

    auto flush = [&]() {
        if (current.problem_indices.empty()) return;
        current.transpose_volume = current_vol;
        result.max_batch_transpose_volume = std::max(result.max_batch_transpose_volume, current_vol);
        result.max_batch_expanded_problems = std::max(result.max_batch_expanded_problems, current_expanded);
        result.batches.push_back(std::move(current));
        current = TransposeBatch{};
        ++batch_gen;
        current_vol = 0;
        current_expanded = 0;
    };

    // Single batch when cap disabled
    if (buffer_cap_elements <= 0) {
        for (int i = 0; i < num_problems; ++i)
            add_problem(problem_order[i]);
        flush();
        return result;
    }

    for (int i = 0; i < num_problems; ++i) {
        int idx = problem_order[i];
        const auto& prob = large.problems[idx];
        int64_t added = 0;
        if (gen_A[prob.block_idx_A] != batch_gen)
            added += A_gpu.block_volumes[prob.block_idx_A];
        if (gen_B[prob.block_idx_B] != batch_gen)
            added += B_gpu.block_volumes[prob.block_idx_B];

        if (!current.problem_indices.empty() && current_vol + added > buffer_cap_elements) {
            flush();
            added = A_gpu.block_volumes[prob.block_idx_A]
                  + B_gpu.block_volumes[prob.block_idx_B];
        }

        add_problem(idx);
    }
    flush();

    return result;
}

// Naive partition: greedy scan in original problem order.
template <typename T>
BatchPartitionResult partition_large_into_batches(
    const LargeBlockWork<T>& large,
    const GPUSparseInput<T>& A_gpu, const GPUSparseInput<T>& B_gpu,
    int64_t buffer_cap_elements)
{
    const int n = static_cast<int>(large.problems.size());
    std::vector<int> identity(n);
    std::iota(identity.begin(), identity.end(), 0);
    return greedy_batch_scan(large, A_gpu, B_gpu, identity.data(), n, buffer_cap_elements);
}

// Smart partition: reorders problems for better A/B locality and K-group separation
// before running the greedy capacity scan.
template <typename T>
BatchPartitionResult partition_large_smart(
    const LargeBlockWork<T>& large,
    const GPUSparseInput<T>& A_gpu, const GPUSparseInput<T>& B_gpu,
    const AccumulationMap& accum_map,
    int64_t buffer_cap_elements)
{
    if (large.empty() || buffer_cap_elements <= 0)
        return partition_large_into_batches(large, A_gpu, B_gpu, buffer_cap_elements);

    const int n = static_cast<int>(large.problems.size());

    // Build K-groups (output_block_idx -> problem indices) and assign K-rank
    std::vector<int64_t> ob_keys(n);
    for (int i = 0; i < n; ++i)
        ob_keys[i] = accum_map.h_pair_output_block_idx[large.problems[i].pair_idx];
    auto k_groups = group_by_int64_key(ob_keys.data(), n);

    std::vector<int> k_rank(n, 0);
    for (int g = 0; g < k_groups.num_groups; ++g)
        for (int r = 0; r < k_groups.group_size(g); ++r)
            k_rank[k_groups.group_items[k_groups.group_offsets[g] + r]] = r;

    // Compute redundancy costs to choose sort direction.
    // cost_sort_by_X = redundancy of the *other* pool when primary sort is X.
    // Dense per-block partner counts replace nested hash sets: each (A,B)
    // block pair appears in at most one problem, so the partner count equals
    // the distinct-partner count (and this only biases a sort heuristic).
    std::vector<int32_t> deg_A(static_cast<size_t>(A_gpu.num_blocks), 0);
    std::vector<int32_t> deg_B(static_cast<size_t>(B_gpu.num_blocks), 0);
    for (int i = 0; i < n; ++i) {
        deg_A[large.problems[i].block_idx_A]++;
        deg_B[large.problems[i].block_idx_B]++;
    }

    int64_t cost_sort_by_A = 0;
    for (int64_t b = 0; b < B_gpu.num_blocks; ++b)
        if (deg_B[b] > 1)
            cost_sort_by_A += int64_t(B_gpu.block_volumes[b]) * (deg_B[b] - 1);

    int64_t cost_sort_by_B = 0;
    for (int64_t a = 0; a < A_gpu.num_blocks; ++a)
        if (deg_A[a] > 1)
            cost_sort_by_B += int64_t(A_gpu.block_volumes[a]) * (deg_A[a] - 1);

    bool sort_by_A = (cost_sort_by_A <= cost_sort_by_B);

    // Sort: (k_rank, primary_block_idx, secondary_block_idx)
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int i, int j) {
        if (k_rank[i] != k_rank[j]) return k_rank[i] < k_rank[j];
        const auto& pi = large.problems[i];
        const auto& pj = large.problems[j];
        int32_t pri_i = sort_by_A ? pi.block_idx_A : pi.block_idx_B;
        int32_t pri_j = sort_by_A ? pj.block_idx_A : pj.block_idx_B;
        if (pri_i != pri_j) return pri_i < pri_j;
        int32_t sec_i = sort_by_A ? pi.block_idx_B : pi.block_idx_A;
        int32_t sec_j = sort_by_A ? pj.block_idx_B : pj.block_idx_A;
        return sec_i < sec_j;
    });

    // Greedy capacity scan on reordered problems
    auto result = greedy_batch_scan(large, A_gpu, B_gpu, order.data(), n, buffer_cap_elements);

    // Hand the output-block CSR to compute_pair_separation (same problem
    // indexing — the scan reorders via batch indices, not the problems vector).
    result.ob_groups = std::move(k_groups);
    result.has_ob_groups = true;

    return result;
}

} // namespace bsptc
