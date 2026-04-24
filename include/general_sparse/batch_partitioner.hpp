#pragma once

#include "pipeline_descs.hpp"
#include "gpu_sparse_input.hpp"
#include "gpu_accumulator.cuh"
#include "ob_grouping.hpp"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <iomanip>

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

struct BatchPartitionStats {
    int num_batches = 0;
    int64_t max_batch_problems = 0;
    int64_t max_batch_expanded = 0;
    int64_t max_batch_transpose_vol = 0;

    int64_t total_A_transpose_vol = 0;
    int64_t total_B_transpose_vol = 0;
    int64_t unique_A_vol = 0;
    int64_t unique_B_vol = 0;

    int64_t num_output_blocks_total = 0;
    int64_t num_output_blocks_single_pair = 0;
    int64_t num_output_blocks_multi_pair = 0;
    int64_t num_output_blocks_separated = 0;
    int64_t num_output_blocks_conflicted = 0;

    bool sort_by_A = false;
    int64_t cost_sort_by_A = 0;
    int64_t cost_sort_by_B = 0;

    double analysis_ms = 0;
};

struct BatchPartitionResult {
    std::vector<TransposeBatch> batches;
    int64_t max_batch_transpose_volume = 0;
    int64_t max_batch_expanded_problems = 0;
    BatchPartitionStats stats;

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

    // Group problems by output block (sort-based, CSR result)
    std::vector<int64_t> ob_keys(n);
    for (int i = 0; i < n; ++i)
        ob_keys[i] = accum_map.h_pair_output_block_idx[large.problems[i].pair_idx];
    auto ob_groups = group_by_int64_key(ob_keys.data(), n);

    for (int g = 0; g < ob_groups.num_groups; ++g) {
        if (ob_groups.group_size(g) <= 1) continue;
        const int* begin = ob_groups.group_begin(g);
        const int* end = ob_groups.group_end(g);
        std::unordered_set<int> batches_used;
        bool separated = true;
        for (const int* it = begin; it != end; ++it) {
            if (!batches_used.insert(prob_batch[*it]).second) {
                separated = false;
                break;
            }
        }
        if (separated) {
            for (const int* it = begin; it != end; ++it)
                result.pair_separated[*it] = 1;
        }
    }
}

inline void print_batch_partition_stats(const BatchPartitionStats& s) {
    std::cout << "  Partition stats (" << std::fixed << std::setprecision(2)
              << s.analysis_ms << " ms):\n";
    std::cout << "    Batches: " << s.num_batches
              << ", max_problems/batch: " << s.max_batch_problems
              << ", max_expanded/batch: " << s.max_batch_expanded << "\n";

    double A_ratio = s.unique_A_vol > 0 ? double(s.total_A_transpose_vol) / double(s.unique_A_vol) : 1.0;
    double B_ratio = s.unique_B_vol > 0 ? double(s.total_B_transpose_vol) / double(s.unique_B_vol) : 1.0;
    std::cout << "    A redundancy: " << A_ratio
              << "x (" << s.total_A_transpose_vol << "/" << s.unique_A_vol << ")"
              << ", B redundancy: " << B_ratio
              << "x (" << s.total_B_transpose_vol << "/" << s.unique_B_vol << ")\n";
    std::cout << "    Sort key: " << (s.sort_by_A ? "A" : "B")
              << " (cost_by_A=" << s.cost_sort_by_A << ", cost_by_B=" << s.cost_sort_by_B << ")\n";
    std::cout << "    Output blocks: " << s.num_output_blocks_total
              << " (single=" << s.num_output_blocks_single_pair
              << ", multi=" << s.num_output_blocks_multi_pair
              << ", separated=" << s.num_output_blocks_separated
              << ", conflicted=" << s.num_output_blocks_conflicted << ")\n";
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

    // Helper: add a problem to the current batch, tracking unique blocks and pack offsets.
    // seen_A/B map offset → index in unique_blocks_A/B for dedup.
    TransposeBatch current;
    std::unordered_map<int64_t, int> seen_A, seen_B;
    int64_t current_vol = 0, current_expanded = 0;

    auto add_problem = [&](int idx) {
        const auto& prob = large.problems[idx];
        current.problem_indices.push_back(idx);
        current_expanded += prob.batch_size;

        auto [itA, newA] = seen_A.emplace(prob.offset_A, static_cast<int>(current.unique_blocks_A.size()));
        if (newA) {
            current.unique_blocks_A.push_back(prob.block_idx_A);
            current.pack_offset_A.push_back(current_vol);
            current_vol += A_gpu.block_volumes[prob.block_idx_A];
        }
        current.local_A.push_back(itA->second);

        auto [itB, newB] = seen_B.emplace(prob.offset_B,
            static_cast<int>(current.unique_blocks_B.size()));
        if (newB) {
            current.unique_blocks_B.push_back(prob.block_idx_B);
            current.pack_offset_B.push_back(current_vol);
            current_vol += B_gpu.block_volumes[prob.block_idx_B];
        }
        current.local_B.push_back(itB->second);
    };

    auto flush = [&]() {
        if (current.problem_indices.empty()) return;
        current.transpose_volume = current_vol;
        result.max_batch_transpose_volume = std::max(result.max_batch_transpose_volume, current_vol);
        result.max_batch_expanded_problems = std::max(result.max_batch_expanded_problems, current_expanded);
        result.batches.push_back(std::move(current));
        current = TransposeBatch{};
        seen_A.clear();
        seen_B.clear();
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
        if (!seen_A.count(prob.offset_A))
            added += A_gpu.block_volumes[prob.block_idx_A];
        if (!seen_B.count(prob.offset_B))
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

// Compute stats for an existing partition result.
template <typename T>
void compute_batch_partition_stats(
    BatchPartitionResult& result,
    const LargeBlockWork<T>& large,
    const GPUSparseInput<T>& A_gpu, const GPUSparseInput<T>& B_gpu,
    const AccumulationMap& accum_map)
{
    auto& s = result.stats;
    s.num_batches = static_cast<int>(result.batches.size());
    s.max_batch_transpose_vol = result.max_batch_transpose_volume;
    s.max_batch_expanded = result.max_batch_expanded_problems;

    // Unique A/B volumes (single-batch baseline) and per-batch redundancy
    std::unordered_set<int64_t> global_A, global_B;
    for (const auto& prob : large.problems) {
        if (global_A.insert(prob.offset_A).second)
            s.unique_A_vol += A_gpu.block_volumes[prob.block_idx_A];
        if (global_B.insert(prob.offset_B).second)
            s.unique_B_vol += B_gpu.block_volumes[prob.block_idx_B];
    }
    for (const auto& batch : result.batches) {
        s.max_batch_problems = std::max(s.max_batch_problems,
                                        static_cast<int64_t>(batch.problem_indices.size()));
        std::unordered_set<int64_t> seen_A, seen_B;
        for (int pi : batch.problem_indices) {
            const auto& prob = large.problems[pi];
            if (seen_A.insert(prob.offset_A).second)
                s.total_A_transpose_vol += A_gpu.block_volumes[prob.block_idx_A];
            if (seen_B.insert(prob.offset_B).second)
                s.total_B_transpose_vol += B_gpu.block_volumes[prob.block_idx_B];
        }
    }

    // K-splitting: group problems by output block, check batch separation.
    const int n = static_cast<int>(large.problems.size());
    std::vector<int64_t> ob_keys(n);
    std::vector<int> prob_to_batch(n, 0);
    for (int bi = 0; bi < static_cast<int>(result.batches.size()); ++bi)
        for (int pi : result.batches[bi].problem_indices)
            prob_to_batch[pi] = bi;
    for (int i = 0; i < n; ++i)
        ob_keys[i] = accum_map.h_pair_output_block_idx[large.problems[i].pair_idx];

    auto ob_groups = group_by_int64_key(ob_keys.data(), n);
    s.num_output_blocks_total = ob_groups.num_groups;
    for (int g = 0; g < ob_groups.num_groups; ++g) {
        if (ob_groups.group_size(g) == 1) {
            s.num_output_blocks_single_pair++;
        } else {
            s.num_output_blocks_multi_pair++;
            const int* begin = ob_groups.group_begin(g);
            const int* end = ob_groups.group_end(g);
            std::unordered_set<int> unique_batches;
            for (const int* it = begin; it != end; ++it)
                unique_batches.insert(prob_to_batch[*it]);
            if (static_cast<int>(unique_batches.size()) == ob_groups.group_size(g))
                s.num_output_blocks_separated++;
            else
                s.num_output_blocks_conflicted++;
        }
    }
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

    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

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
    std::unordered_map<int32_t, std::unordered_set<int32_t>> A_to_Bs, B_to_As;
    for (int i = 0; i < n; ++i) {
        const auto& p = large.problems[i];
        A_to_Bs[p.block_idx_A].insert(p.block_idx_B);
        B_to_As[p.block_idx_B].insert(p.block_idx_A);
    }

    int64_t cost_sort_by_A = 0;
    for (const auto& [b, a_set] : B_to_As)
        if (a_set.size() > 1)
            cost_sort_by_A += int64_t(B_gpu.block_volumes[b]) * (int64_t(a_set.size()) - 1);

    int64_t cost_sort_by_B = 0;
    for (const auto& [a, b_set] : A_to_Bs)
        if (b_set.size() > 1)
            cost_sort_by_B += int64_t(A_gpu.block_volumes[a]) * (int64_t(b_set.size()) - 1);

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
    result.stats.analysis_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    result.stats.sort_by_A = sort_by_A;
    result.stats.cost_sort_by_A = cost_sort_by_A;
    result.stats.cost_sort_by_B = cost_sort_by_B;

    return result;
}

} // namespace bsptc
