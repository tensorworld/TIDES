#pragma once

#include "coord_key.hpp"
#include "gpu_sparse_device.cuh"
#include "gpu_sparse_input.hpp"
#include "hash_tables.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

// GPU-based accumulation for block sparse contraction results.
// Replaces CPU accumulation (GPU->CPU copy, std::map, CPU->GPU copy) with
// a GPU kernel that uses atomicAdd to accumulate in-place.
namespace bsptc {

// Per-output-block tier presence flags (bitmask stored in h_ob_tier_flags).
static constexpr uint8_t kOBHasSmall    = 0x01;  // has warp-tier pairs
static constexpr uint8_t kOBHasMedium   = 0x02;  // has volume-tier pairs
static constexpr uint8_t kOBHasLarge    = 0x04;  // has CUTLASS-tier pairs
static constexpr uint8_t kOBMultiSmall  = 0x08;  // >1 small pair for this output block
static constexpr uint8_t kOBMultiMedium = 0x10;  // >1 medium pair for this output block
static constexpr uint8_t kOBMultiLarge  = 0x20;  // >1 large pair for this output block

// Pre-computed output block structure.
// Built once from A×B pair info; reusable across iterations when the
// sparsity pattern (block coordinates) doesn't change.
struct OutputBlockMap {
    std::vector<int32_t> block_coords;   // [num_blocks * rank] flat, internal order
    std::vector<int64_t> block_volumes;  // [num_blocks]
    std::vector<int64_t> block_offsets;  // [num_blocks] cumulative
    int64_t num_blocks = 0;
    int64_t total_elements = 0;
    dim_t rank = 0;

    // Lookup: output coord → block index (reuses CoordKeyDynamic from coord_key.hpp)
    std::unordered_map<CoordKeyDynamic, int32_t, CoordKeyDynamicHash> coord_to_idx;

    // Convert to cuTENSOR-compatible output-label-order coords (int64_t).
    // internal_to_output[i] = position in output label order for internal dim i.
    // Returns sorted, deduplicated coords (matching compute_output_block_coords).
    std::vector<std::vector<int64_t>> to_output_order(
        const int32_t* internal_to_output) const
    {
        std::vector<std::vector<int64_t>> result(static_cast<size_t>(num_blocks));
        for (int64_t b = 0; b < num_blocks; ++b) {
            result[static_cast<size_t>(b)].resize(static_cast<size_t>(rank));
            const int32_t* src = &block_coords[b * rank];
            for (int32_t d = 0; d < rank; ++d) {
                result[static_cast<size_t>(b)][static_cast<size_t>(
                    internal_to_output[d])] = src[d];
            }
        }
        std::sort(result.begin(), result.end());
        return result;
    }
};

// Extract the output block coordinate key for a pair (batch + free_A + free_B).
template <typename T>
inline CoordKeyDynamic extract_output_key(
    const BlockPairInfo& pair,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const ContractionConfig& config)
{
    CoordKeyDynamic key{};
    key.len = config.output_rank;
    int32_t pos = 0;
    for (int32_t b = 0; b < config.num_batch; ++b)
        key.coords[pos++] = A_gpu.block_coords[pair.block_idx_A * A_gpu.rank + config.batch_pos_A[b]];
    for (int32_t f = 0; f < config.num_free_A; ++f)
        key.coords[pos++] = A_gpu.block_coords[pair.block_idx_A * A_gpu.rank + config.free_pos_A[f]];
    for (int32_t f = 0; f < config.num_free_B; ++f)
        key.coords[pos++] = B_gpu.block_coords[pair.block_idx_B * B_gpu.rank + config.free_pos_B[f]];
    return key;
}

// Compute the output block structure from matched pairs.
// O(num_pairs) via hash map — no sort. Call once, reuse across iterations.
template <typename T>
OutputBlockMap compute_output_blocks(
    const BlockPairInfo* pairs_host,
    int64_t num_pairs,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const ContractionConfig& config)
{
    OutputBlockMap obm;
    obm.rank = config.output_rank;
    const int32_t rank = config.output_rank;

    obm.coord_to_idx.reserve(static_cast<size_t>(num_pairs / 4 + 1));

    for (int64_t i = 0; i < num_pairs; ++i) {
        auto key = extract_output_key(pairs_host[i], A_gpu, B_gpu, config);

        auto it = obm.coord_to_idx.find(key);
        if (it == obm.coord_to_idx.end()) {
            int32_t idx = static_cast<int32_t>(obm.num_blocks);
            obm.coord_to_idx[key] = idx;
            for (int32_t d = 0; d < rank; ++d)
                obm.block_coords.push_back(key.coords[d]);
            obm.block_volumes.push_back(pairs_host[i].output_volume);
            obm.num_blocks++;
        }
    }

    // Cumulative offsets
    obm.block_offsets.resize(static_cast<size_t>(obm.num_blocks));
    int64_t offset = 0;
    for (int64_t i = 0; i < obm.num_blocks; ++i) {
        obm.block_offsets[static_cast<size_t>(i)] = offset;
        offset += obm.block_volumes[static_cast<size_t>(i)];
    }
    obm.total_elements = offset;
    return obm;
}

// Accumulation map: maps each pair to its destination offset in the final output.
// Not templated on T (only stores offsets/indices).
// Owns its arrays: move-only, destructor releases host and device memory.
struct AccumulationMap {
    // Host arrays (raw new[] — deliberately uninitialized; every entry is
    // written by the fill passes, so no vector zero-init cost at scale)
    int64_t* h_dest_offsets = nullptr;          // [num_pairs] per-pair dest offset in final output
    int64_t* h_pair_output_block_idx = nullptr; // [num_pairs] output block index for this pair
    int32_t* h_output_block_coords = nullptr;   // [num_output_blocks * output_rank] flat
    int64_t* h_output_block_offsets = nullptr;  // [num_output_blocks] offset of each output block
    int64_t* h_output_block_volumes = nullptr;  // [num_output_blocks] volume of each output block
    int64_t* h_output_pair_offsets = nullptr;   // [num_output_blocks + 1] flattened pair list offsets
    int64_t* h_output_pair_indices = nullptr;   // [num_pairs] flattened pair indices by output block
    uint8_t* h_ob_tier_flags = nullptr;         // [num_output_blocks] per-OB tier presence bitfield

    int64_t total_output_elements = 0;
    int64_t num_output_blocks = 0;
    int64_t num_pairs = 0;
    dim_t output_rank = 0;

    AccumulationMap() = default;
    AccumulationMap(const AccumulationMap&) = delete;
    AccumulationMap& operator=(const AccumulationMap&) = delete;
    AccumulationMap(AccumulationMap&& o) noexcept { swap(o); }
    AccumulationMap& operator=(AccumulationMap&& o) noexcept {
        if (this != &o) { release(); swap(o); }
        return *this;
    }
    ~AccumulationMap() { release(); }

    void swap(AccumulationMap& o) noexcept {
        std::swap(h_dest_offsets, o.h_dest_offsets);
        std::swap(h_pair_output_block_idx, o.h_pair_output_block_idx);
        std::swap(h_output_block_coords, o.h_output_block_coords);
        std::swap(h_output_block_offsets, o.h_output_block_offsets);
        std::swap(h_output_block_volumes, o.h_output_block_volumes);
        std::swap(h_output_pair_offsets, o.h_output_pair_offsets);
        std::swap(h_output_pair_indices, o.h_output_pair_indices);
        std::swap(h_ob_tier_flags, o.h_ob_tier_flags);
        std::swap(total_output_elements, o.total_output_elements);
        std::swap(num_output_blocks, o.num_output_blocks);
        std::swap(num_pairs, o.num_pairs);
        std::swap(output_rank, o.output_rank);
    }

    void release() noexcept {
        delete[] h_dest_offsets;          h_dest_offsets = nullptr;
        delete[] h_pair_output_block_idx; h_pair_output_block_idx = nullptr;
        delete[] h_output_block_coords;   h_output_block_coords = nullptr;
        delete[] h_output_block_offsets;  h_output_block_offsets = nullptr;
        delete[] h_output_block_volumes;  h_output_block_volumes = nullptr;
        delete[] h_output_pair_offsets;   h_output_pair_offsets = nullptr;
        delete[] h_output_pair_indices;   h_output_pair_indices = nullptr;
        delete[] h_ob_tier_flags;         h_ob_tier_flags = nullptr;
    }
};

// Reduction plan for temp-buffer pairs (large + medium):
// - direct-write singleton CUTLASS outputs to final output
// - reduce multi-write output blocks with a no-atomic kernel
// Owns its arrays: move-only, destructor releases host and device memory.
struct ReductionPlan {
    // Host flags per original pair index
    uint8_t* h_direct_write = nullptr;  // [num_pairs], 1 if CUTLASS singleton

    // Host reduction groups (output blocks with >1 temp-buffer pairs)
    int64_t num_reduce_groups = 0;
    int64_t* h_reduce_dest_offsets = nullptr; // [num_reduce_groups]
    int64_t* h_reduce_volumes = nullptr;      // [num_reduce_groups]
    int64_t* h_reduce_src_offsets = nullptr;  // [num_reduce_src]
    int64_t* h_reduce_src_ptrs = nullptr;     // [num_reduce_groups + 1]

    // Device reduction groups
    int64_t* d_reduce_dest_offsets = nullptr;
    int64_t* d_reduce_volumes = nullptr;
    int64_t* d_reduce_src_offsets = nullptr;
    int64_t* d_reduce_src_ptrs = nullptr;

    ReductionPlan() = default;
    ReductionPlan(const ReductionPlan&) = delete;
    ReductionPlan& operator=(const ReductionPlan&) = delete;
    ReductionPlan(ReductionPlan&& o) noexcept { swap(o); }
    ReductionPlan& operator=(ReductionPlan&& o) noexcept {
        if (this != &o) { release(); swap(o); }
        return *this;
    }
    ~ReductionPlan() { release(); }

    void swap(ReductionPlan& o) noexcept {
        std::swap(h_direct_write, o.h_direct_write);
        std::swap(num_reduce_groups, o.num_reduce_groups);
        std::swap(h_reduce_dest_offsets, o.h_reduce_dest_offsets);
        std::swap(h_reduce_volumes, o.h_reduce_volumes);
        std::swap(h_reduce_src_offsets, o.h_reduce_src_offsets);
        std::swap(h_reduce_src_ptrs, o.h_reduce_src_ptrs);
        std::swap(d_reduce_dest_offsets, o.d_reduce_dest_offsets);
        std::swap(d_reduce_volumes, o.d_reduce_volumes);
        std::swap(d_reduce_src_offsets, o.d_reduce_src_offsets);
        std::swap(d_reduce_src_ptrs, o.d_reduce_src_ptrs);
    }

    void release() noexcept {
        delete[] h_direct_write;         h_direct_write = nullptr;
        delete[] h_reduce_dest_offsets;  h_reduce_dest_offsets = nullptr;
        delete[] h_reduce_volumes;       h_reduce_volumes = nullptr;
        delete[] h_reduce_src_offsets;   h_reduce_src_offsets = nullptr;
        delete[] h_reduce_src_ptrs;      h_reduce_src_ptrs = nullptr;
        if (d_reduce_dest_offsets) { cudaFree(d_reduce_dest_offsets); d_reduce_dest_offsets = nullptr; }
        if (d_reduce_volumes)      { cudaFree(d_reduce_volumes);      d_reduce_volumes = nullptr; }
        if (d_reduce_src_offsets)  { cudaFree(d_reduce_src_offsets);  d_reduce_src_offsets = nullptr; }
        if (d_reduce_src_ptrs)     { cudaFree(d_reduce_src_ptrs);     d_reduce_src_ptrs = nullptr; }
    }
};

// Allocate the host arrays of an AccumulationMap sized for
// num_pairs/num_output_blocks/output_rank (all already set on `map`).
inline void alloc_accumulation_map_arrays(AccumulationMap& map, bool want_tier_flags) {
    const int64_t np = map.num_pairs;
    const int64_t nob = map.num_output_blocks;
    map.h_dest_offsets = new int64_t[np];
    map.h_pair_output_block_idx = new int64_t[np];
    map.h_output_block_coords = new int32_t[nob * map.output_rank];
    map.h_output_block_offsets = new int64_t[nob];
    map.h_output_block_volumes = new int64_t[nob];
    map.h_output_pair_offsets = new int64_t[nob + 1];
    map.h_output_pair_indices = new int64_t[np];
    if (want_tier_flags) {
        map.h_ob_tier_flags = new uint8_t[nob];
        std::memset(map.h_ob_tier_flags, 0, static_cast<size_t>(nob) * sizeof(uint8_t));
    }
}

// Shared tail of both compute_accumulation_map overloads. Requires
// h_pair_output_block_idx and the block-level arrays (coords, offsets,
// volumes) to be filled; derives dest offsets, the grouped pair-index
// lists, and (when work_tier is given) the per-OB tier flags.
inline void fill_accumulation_pair_arrays(AccumulationMap& map, const uint8_t* work_tier) {
    const int64_t np = map.num_pairs;
    const int64_t nob = map.num_output_blocks;

    std::vector<int32_t> pairs_per_block(static_cast<size_t>(nob), 0);
    for (int64_t i = 0; i < np; ++i)
        pairs_per_block[static_cast<size_t>(map.h_pair_output_block_idx[i])]++;

    int64_t pair_cursor = 0;
    for (int64_t ob = 0; ob < nob; ++ob) {
        map.h_output_pair_offsets[ob] = pair_cursor;
        pair_cursor += pairs_per_block[static_cast<size_t>(ob)];
    }
    map.h_output_pair_offsets[nob] = pair_cursor;

    std::vector<int32_t> cursor(static_cast<size_t>(nob), 0);
    for (int64_t i = 0; i < np; ++i) {
        const int64_t ob = map.h_pair_output_block_idx[i];
        map.h_dest_offsets[i] = map.h_output_block_offsets[ob];
        const int64_t slot = map.h_output_pair_offsets[ob] + cursor[static_cast<size_t>(ob)]++;
        map.h_output_pair_indices[slot] = i;
        if (work_tier) {
            const uint8_t t = work_tier[i];
            uint8_t& flags = map.h_ob_tier_flags[ob];
            const uint8_t has = static_cast<uint8_t>(kOBHasSmall << t);
            flags |= (flags & has) ? static_cast<uint8_t>(has << 3) : has;
        }
    }
}

// Compute the accumulation map from host-side pair info.
// Groups pairs by output block coordinate via O(n) hash grouping
// (group_coord_rows, gtl Swiss table in the gtl-isolated hash_tables TU).
// Output blocks are ordered by first occurrence in pair order — the same
// contract as the precomputed-OutputBlockMap path below.
template <typename T>
AccumulationMap compute_accumulation_map(
    const BlockPairInfo* pairs_host,
    int64_t num_pairs,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const ContractionConfig& config,
    const uint8_t* work_tier = nullptr)
{
    AccumulationMap map;
    map.num_pairs = num_pairs;
    map.output_rank = config.output_rank;
    map.h_ob_tier_flags = nullptr;

    const int32_t rank = config.output_rank;

    // Flat coordinate storage: [num_pairs * rank], tightly packed
    std::vector<int32_t> flat_coords(num_pairs * rank);
    for (int64_t i = 0; i < num_pairs; ++i) {
        int32_t* coord_out = flat_coords.data() + i * rank;
        int32_t pos = 0;
        for (int32_t b = 0; b < config.num_batch; ++b)
            coord_out[pos++] = A_gpu.block_coords[pairs_host[i].block_idx_A * A_gpu.rank + config.batch_pos_A[b]];
        for (int32_t f = 0; f < config.num_free_A; ++f)
            coord_out[pos++] = A_gpu.block_coords[pairs_host[i].block_idx_A * A_gpu.rank + config.free_pos_A[f]];
        for (int32_t f = 0; f < config.num_free_B; ++f)
            coord_out[pos++] = B_gpu.block_coords[pairs_host[i].block_idx_B * B_gpu.rank + config.free_pos_B[f]];
    }

    std::vector<int32_t> group_of(static_cast<size_t>(num_pairs));
    const int64_t num_output_blocks = group_coord_rows(
        flat_coords.data(), num_pairs, rank, group_of.data());
    map.num_output_blocks = num_output_blocks;

    alloc_accumulation_map_arrays(map, work_tier != nullptr);

    // Assign pairs to blocks; remember each block's first pair (coords + volume)
    std::vector<int32_t> first_pair(static_cast<size_t>(num_output_blocks), -1);
    for (int64_t i = 0; i < num_pairs; ++i) {
        const int32_t ob = group_of[static_cast<size_t>(i)];
        map.h_pair_output_block_idx[i] = ob;
        if (first_pair[static_cast<size_t>(ob)] < 0)
            first_pair[static_cast<size_t>(ob)] = static_cast<int32_t>(i);
    }

    // Block-level arrays + offset prefix sum
    int64_t current_offset = 0;
    for (int64_t ob = 0; ob < num_output_blocks; ++ob) {
        const int32_t fp = first_pair[static_cast<size_t>(ob)];
        std::memcpy(map.h_output_block_coords + ob * rank,
                    flat_coords.data() + static_cast<int64_t>(fp) * rank,
                    static_cast<size_t>(rank) * sizeof(int32_t));
        const int64_t volume = pairs_host[fp].output_volume;
        map.h_output_block_offsets[ob] = current_offset;
        map.h_output_block_volumes[ob] = volume;
        current_offset += volume;
    }
    map.total_output_elements = current_offset;

    fill_accumulation_pair_arrays(map, work_tier);
    return map;
}

// Fast O(n) accumulation map construction using pre-computed OutputBlockMap.
// Replaces the O(n log n) sort path when output block structure is known.
template <typename T>
AccumulationMap compute_accumulation_map(
    const BlockPairInfo* pairs_host,
    int64_t num_pairs,
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const ContractionConfig& config,
    const OutputBlockMap& obm,
    const uint8_t* work_tier = nullptr)
{
    AccumulationMap map;
    map.num_pairs = num_pairs;
    map.output_rank = config.output_rank;
    map.h_ob_tier_flags = nullptr;

    const int32_t rank = config.output_rank;
    const int64_t nob = obm.num_blocks;
    map.num_output_blocks = nob;

    alloc_accumulation_map_arrays(map, work_tier != nullptr);

    // Copy block-level data from OutputBlockMap
    std::memcpy(map.h_output_block_coords, obm.block_coords.data(),
                static_cast<size_t>(nob * rank) * sizeof(int32_t));
    std::memcpy(map.h_output_block_offsets, obm.block_offsets.data(),
                static_cast<size_t>(nob) * sizeof(int64_t));
    std::memcpy(map.h_output_block_volumes, obm.block_volumes.data(),
                static_cast<size_t>(nob) * sizeof(int64_t));

    // Assign each pair to its output block via hash lookup.
    // Per-pair find into the nob-entry map beats O(n) Swiss-table grouping
    // here: the map is small enough to stay cache-resident, while grouping
    // builds an num_pairs-entry table (measured slower at 524k pairs).
    for (int64_t i = 0; i < num_pairs; ++i) {
        auto key = extract_output_key(pairs_host[i], A_gpu, B_gpu, config);
        auto it = obm.coord_to_idx.find(key);
        if (it == obm.coord_to_idx.end())
            throw std::runtime_error(
                "compute_accumulation_map: a block pair produces an output "
                "block absent from the provided OutputBlockMap (caller output "
                "metadata is inconsistent with the inputs)");
        map.h_pair_output_block_idx[i] = it->second;
    }

    map.total_output_elements = obm.total_elements;

    fill_accumulation_pair_arrays(map, work_tier);
    return map;
}

// Direct-write eligibility, computed ONCE per plan: a temp-tier (CUTLASS)
// pair may bypass the temp buffer when it is the sole temp-tier writer of its
// output block AND the block is not mixed (kOBHasLarge & kOBHasMedium —
// volume pairs write d_output concurrently). Consumed by both
// plan_temp_buffer_offsets (skip temp slots) and build_reduction_plan
// (h_direct_write + reduce groups), which previously each recomputed it.
// pair_tier: 0 = warp/volume (d_output), nonzero = temp tier.
inline std::vector<uint8_t> compute_direct_write_eligibility(
    const AccumulationMap& map,
    const std::vector<uint8_t>& pair_tier)
{
    const size_t np = pair_tier.size();
    const size_t nob = static_cast<size_t>(map.num_output_blocks);
    std::vector<uint8_t> is_direct(np, 0);
    if (np == 0 || nob == 0) return is_direct;

    std::vector<int64_t> tb_count(nob, 0);
    for (size_t p = 0; p < np; ++p)
        if (pair_tier[p] != 0)
            tb_count[static_cast<size_t>(map.h_pair_output_block_idx[p])]++;

    for (size_t ob = 0; ob < nob; ++ob) {
        if (tb_count[ob] != 1) continue;
        if (map.h_ob_tier_flags &&
            (map.h_ob_tier_flags[ob] & (kOBHasLarge | kOBHasMedium)) ==
            (kOBHasLarge | kOBHasMedium)) continue;
        const int64_t begin = map.h_output_pair_offsets[ob];
        const int64_t end = map.h_output_pair_offsets[ob + 1];
        for (int64_t i = begin; i < end; ++i) {
            const int64_t pi = map.h_output_pair_indices[i];
            if (pair_tier[static_cast<size_t>(pi)] != 0) {
                is_direct[static_cast<size_t>(pi)] = 1;
                break;
            }
        }
    }
    return is_direct;
}

// Build reduction plan for the CUTLASS tier (only tier using temp buffer):
// - `h_direct_write[p] == 1` means pair p can write directly to final output
// - output blocks with >1 temp-buffer pairs become no-atomic reduction groups
// pair_tier: 0 = warp/volume (write to d_output, no temp buffer),
//            2 = CUTLASS (temp buffer, eligible for direct-write singleton).
// is_direct: precomputed by compute_direct_write_eligibility (same pair_tier).
inline ReductionPlan build_reduction_plan(
    const AccumulationMap& map,
    const std::vector<uint8_t>& pair_tier,
    const std::vector<uint8_t>& is_direct,
    const BlockPairInfo* pairs_host)
{
    ReductionPlan plan;

    if (map.num_pairs > 0) {
        plan.h_direct_write = new uint8_t[map.num_pairs];
        std::memset(plan.h_direct_write, 0, map.num_pairs * sizeof(uint8_t));
    }

    if (map.num_pairs == 0 || map.num_output_blocks == 0 ||
        pair_tier.size() != static_cast<size_t>(map.num_pairs)) {
        return plan;
    }

    std::memcpy(plan.h_direct_write, is_direct.data(),
                map.num_pairs * sizeof(uint8_t));

    // Collect unique reduce source offsets per output block.
    // With streaming temp-buffer reuse, multiple pairs may share the same
    // temp offset; the reduce kernel must process each unique offset once.
    const size_t nob = static_cast<size_t>(map.num_output_blocks);
    std::vector<std::vector<int64_t>> reduce_srcs(nob);
    for (int64_t p = 0; p < map.num_pairs; ++p) {
        if (pair_tier[p] != 0 && !plan.h_direct_write[p]) {
            const size_t ob = static_cast<size_t>(map.h_pair_output_block_idx[p]);
            int64_t off = pairs_host[p].output_offset;
            auto& vec = reduce_srcs[ob];
            bool dup = false;
            for (int64_t existing : vec) {
                if (existing == off) { dup = true; break; }
            }
            if (!dup) vec.push_back(off);
        }
    }

    // Pass 2: count reduce groups and unique sources.
    int64_t total_reduce_src = 0;
    for (size_t ob = 0; ob < nob; ++ob) {
        if (reduce_srcs[ob].empty()) continue;
        plan.num_reduce_groups++;
        total_reduce_src += static_cast<int64_t>(reduce_srcs[ob].size());
    }

    if (plan.num_reduce_groups == 0) return plan;

    plan.h_reduce_dest_offsets = new int64_t[plan.num_reduce_groups];
    plan.h_reduce_volumes = new int64_t[plan.num_reduce_groups];
    plan.h_reduce_src_offsets = new int64_t[total_reduce_src];
    plan.h_reduce_src_ptrs = new int64_t[plan.num_reduce_groups + 1];

    int64_t group_idx = 0;
    int64_t src_cursor = 0;

    for (int64_t out_block = 0; out_block < map.num_output_blocks; ++out_block) {
        const auto& srcs = reduce_srcs[static_cast<size_t>(out_block)];
        if (srcs.empty()) continue;

        plan.h_reduce_dest_offsets[group_idx] = map.h_output_block_offsets[out_block];
        plan.h_reduce_volumes[group_idx] = map.h_output_block_volumes[out_block];
        plan.h_reduce_src_ptrs[group_idx] = src_cursor;

        for (int64_t off : srcs) {
            plan.h_reduce_src_offsets[src_cursor++] = off;
        }
        group_idx++;
    }
    plan.h_reduce_src_ptrs[plan.num_reduce_groups] = src_cursor;
    return plan;
}

inline void upload_reduction_plan(ReductionPlan& plan, cudaStream_t stream = 0) {
    if (plan.num_reduce_groups <= 0) return;

    const int64_t num_reduce_src = plan.h_reduce_src_ptrs[plan.num_reduce_groups];
    check_cuda(cudaMalloc(&plan.d_reduce_dest_offsets, plan.num_reduce_groups * sizeof(int64_t)),
               "malloc d_reduce_dest_offsets");
    check_cuda(cudaMalloc(&plan.d_reduce_volumes, plan.num_reduce_groups * sizeof(int64_t)),
               "malloc d_reduce_volumes");
    check_cuda(cudaMalloc(&plan.d_reduce_src_ptrs, (plan.num_reduce_groups + 1) * sizeof(int64_t)),
               "malloc d_reduce_src_ptrs");
    check_cuda(cudaMemcpyAsync(plan.d_reduce_dest_offsets, plan.h_reduce_dest_offsets,
                               plan.num_reduce_groups * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream),
               "memcpy d_reduce_dest_offsets");
    check_cuda(cudaMemcpyAsync(plan.d_reduce_volumes, plan.h_reduce_volumes,
                               plan.num_reduce_groups * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream),
               "memcpy d_reduce_volumes");
    check_cuda(cudaMemcpyAsync(plan.d_reduce_src_ptrs, plan.h_reduce_src_ptrs,
                               (plan.num_reduce_groups + 1) * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream),
               "memcpy d_reduce_src_ptrs");

    if (num_reduce_src > 0) {
        check_cuda(cudaMalloc(&plan.d_reduce_src_offsets, num_reduce_src * sizeof(int64_t)),
                   "malloc d_reduce_src_offsets");
        check_cuda(cudaMemcpyAsync(plan.d_reduce_src_offsets, plan.h_reduce_src_offsets,
                                   num_reduce_src * sizeof(int64_t),
                                   cudaMemcpyHostToDevice, stream),
                   "memcpy d_reduce_src_offsets");
    }
}

// Reduce multi-write outputs without atomics.
// Processes one source at a time to maintain sequential memory access.
// grid.x = reduce groups, grid.y = chunk workers per output block.
template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
__global__ void reduce_noatomic_kernel(
    T* temp_buffer,
    T* final_output,
    const int64_t* reduce_dest_offsets,
    const int64_t* reduce_volumes,
    const int64_t* reduce_src_offsets,
    const int64_t* reduce_src_ptrs,
    int64_t num_reduce_groups)
{
    const int64_t group_idx = blockIdx.x;
    if (group_idx >= num_reduce_groups) return;

    const int64_t dst = reduce_dest_offsets[group_idx];
    const int64_t vol = reduce_volumes[group_idx];
    const int64_t src_begin = reduce_src_ptrs[group_idx];
    const int64_t src_end = reduce_src_ptrs[group_idx + 1];
    const int64_t chunk = static_cast<int64_t>(BLOCK_SIZE) * ELEMENTS_PER_THREAD;

    // Process one source at a time: sequential scan of temp + output
    // avoids L2 thrashing from striding across multiple source offsets.
    for (int64_t s = src_begin; s < src_end; ++s) {
        const int64_t src_off = reduce_src_offsets[s];
        for (int64_t chunk_idx = blockIdx.y;; chunk_idx += gridDim.y) {
            const int64_t start = chunk_idx * chunk;
            if (start >= vol) break;

#pragma unroll
            for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                const int64_t idx = start + static_cast<int64_t>(e) * BLOCK_SIZE + threadIdx.x;
                if (idx < vol) {
                    const int64_t src_idx = src_off + idx;
                    final_output[dst + idx] += temp_buffer[src_idx];
                    temp_buffer[src_idx] = static_cast<T>(0);
                }
            }
        }
    }
}

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
inline void launch_reduce_noatomic(
    T* d_temp_buffer,
    T* d_final_output,
    const ReductionPlan& plan,
    cudaStream_t stream = 0)
{
    if (plan.num_reduce_groups <= 0) return;

    const int64_t chunk = static_cast<int64_t>(BLOCK_SIZE) * ELEMENTS_PER_THREAD;
    int64_t max_chunks = 1;
    for (int64_t g = 0; g < plan.num_reduce_groups; ++g) {
        const int64_t chunks = (plan.h_reduce_volumes[g] + chunk - 1) / chunk;
        max_chunks = std::max(max_chunks, chunks);
    }
    const int grid_y = static_cast<int>(std::min<int64_t>(max_chunks, 65535));
    dim3 grid(static_cast<unsigned int>(plan.num_reduce_groups),
              static_cast<unsigned int>(grid_y), 1);

    reduce_noatomic_kernel<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>
        <<<grid, BLOCK_SIZE, 0, stream>>>(
            d_temp_buffer,
            d_final_output,
            plan.d_reduce_dest_offsets,
            plan.d_reduce_volumes,
            plan.d_reduce_src_offsets,
            plan.d_reduce_src_ptrs,
            plan.num_reduce_groups);
}

} // namespace bsptc
