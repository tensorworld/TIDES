#pragma once

#include "coord_key.hpp"
#include "gpu_sparse_device.cuh"
#include "gpu_sparse_input.hpp"

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

// 16-byte sort key: hash + pair index. Fits in cache for fast sorting.
struct alignas(16) HashEntry {
    uint64_t hash;
    int32_t pair_idx;
};

// FNV-1a hash combine
inline void hash_combine(uint64_t& hash, int32_t val) {
    hash ^= static_cast<uint64_t>(static_cast<uint32_t>(val));
    hash *= 1099511628211ULL;
}

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
struct AccumulationMap {
    // Host arrays (heap-allocated, not std::vector)
    int64_t* h_dest_offsets;          // [num_pairs] per-pair dest offset in final output
    int32_t* h_pair_reuse_counts;     // [num_pairs] total writes to pair's output block
    int64_t* h_pair_output_block_idx; // [num_pairs] output block index for this pair
    int32_t* h_output_block_coords;   // [num_output_blocks * output_rank] flat
    int64_t* h_output_block_offsets;  // [num_output_blocks] offset of each output block
    int64_t* h_output_block_volumes;  // [num_output_blocks] volume of each output block
    int64_t* h_output_pair_offsets;   // [num_output_blocks + 1] flattened pair list offsets
    int64_t* h_output_pair_indices;   // [num_pairs] flattened pair indices by output block
    uint8_t* h_ob_tier_flags;         // [num_output_blocks] per-OB tier presence bitfield

    // Device copy
    int64_t* d_dest_offsets;          // device copy of h_dest_offsets

    int64_t total_output_elements;
    int64_t num_output_blocks;
    int64_t num_pairs;
    dim_t output_rank;
};

// Reduction plan for temp-buffer pairs (large + medium):
// - direct-write singleton CUTLASS outputs to final output
// - reduce multi-write output blocks with a no-atomic kernel
struct ReductionPlan {
    int64_t num_pairs = 0;

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
};

// Compute the accumulation map from host-side pair info.
// Groups pairs by output block coordinate using hash-accelerated sort.
// Output blocks are grouped deterministically but not in lexicographic order.
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
    map.d_dest_offsets = nullptr;
    map.h_ob_tier_flags = nullptr;

    const int32_t rank = config.output_rank;

    // Flat coordinate storage: [num_pairs * rank], tightly packed
    std::vector<int32_t> flat_coords(num_pairs * rank);
    std::vector<HashEntry> entries(num_pairs);

    // Generate coordinates and hashes
    for (int64_t i = 0; i < num_pairs; ++i) {
        int32_t* coord_out = &flat_coords[i * rank];
        uint64_t hash = 14695981039346656037ULL; // FNV offset basis
        int32_t pos = 0;

        for (int32_t b = 0; b < config.num_batch; ++b) {
            int32_t val = A_gpu.block_coords[pairs_host[i].block_idx_A * A_gpu.rank + config.batch_pos_A[b]];
            coord_out[pos++] = val;
            hash_combine(hash, val);
        }
        for (int32_t f = 0; f < config.num_free_A; ++f) {
            int32_t val = A_gpu.block_coords[pairs_host[i].block_idx_A * A_gpu.rank + config.free_pos_A[f]];
            coord_out[pos++] = val;
            hash_combine(hash, val);
        }
        for (int32_t f = 0; f < config.num_free_B; ++f) {
            int32_t val = B_gpu.block_coords[pairs_host[i].block_idx_B * B_gpu.rank + config.free_pos_B[f]];
            coord_out[pos++] = val;
            hash_combine(hash, val);
        }

        entries[i] = {hash, static_cast<int32_t>(i)};
    }

    // Sort by hash; fall back to coordinate comparison on collision
    std::sort(entries.begin(), entries.end(), [&](const HashEntry& a, const HashEntry& b) {
        if (a.hash != b.hash) return a.hash < b.hash;
        const int32_t* c_a = &flat_coords[a.pair_idx * rank];
        const int32_t* c_b = &flat_coords[b.pair_idx * rank];
        for (int32_t d = 0; d < rank; ++d) {
            if (c_a[d] != c_b[d]) return c_a[d] < c_b[d];
        }
        return a.pair_idx < b.pair_idx;
    });

    // Count output blocks (group boundaries)
    int64_t num_output_blocks = 0;
    for (int64_t i = 0; i < num_pairs; ) {
        int64_t j = i + 1;
        while (j < num_pairs && entries[i].hash == entries[j].hash) {
            const int32_t* c_base = &flat_coords[entries[i].pair_idx * rank];
            const int32_t* c_j = &flat_coords[entries[j].pair_idx * rank];
            bool eq = true;
            for (int32_t d = 0; d < rank; ++d) {
                if (c_base[d] != c_j[d]) { eq = false; break; }
            }
            if (!eq) break;
            j++;
        }
        num_output_blocks++;
        i = j;
    }
    map.num_output_blocks = num_output_blocks;

    // Allocate host arrays
    map.h_dest_offsets = new int64_t[num_pairs];
    map.h_pair_reuse_counts = new int32_t[num_pairs];
    map.h_pair_output_block_idx = new int64_t[num_pairs];
    map.h_output_block_coords = new int32_t[num_output_blocks * rank];
    map.h_output_block_offsets = new int64_t[num_output_blocks];
    map.h_output_block_volumes = new int64_t[num_output_blocks];
    map.h_output_pair_offsets = new int64_t[num_output_blocks + 1];
    map.h_output_pair_indices = new int64_t[num_pairs];
    if (work_tier) {
        map.h_ob_tier_flags = new uint8_t[num_output_blocks];
        std::memset(map.h_ob_tier_flags, 0, num_output_blocks * sizeof(uint8_t));
    }

    // Group and reduce: single pass over sorted entries
    int64_t current_offset = 0;
    int64_t block_idx = 0;
    int64_t pair_cursor = 0;

    for (int64_t i = 0; i < num_pairs; ) {
        const int32_t base_pair_idx = entries[i].pair_idx;
        const int32_t* base_coord = &flat_coords[base_pair_idx * rank];

        // Find end of group
        int64_t j = i + 1;
        while (j < num_pairs && entries[i].hash == entries[j].hash) {
            const int32_t* c_j = &flat_coords[entries[j].pair_idx * rank];
            bool eq = true;
            for (int32_t d = 0; d < rank; ++d) {
                if (base_coord[d] != c_j[d]) { eq = false; break; }
            }
            if (!eq) break;
            j++;
        }

        const int32_t reuse = static_cast<int32_t>(j - i);
        const int64_t volume = pairs_host[base_pair_idx].output_volume;

        // Store block coordinate
        for (int32_t d = 0; d < rank; ++d) {
            map.h_output_block_coords[block_idx * rank + d] = base_coord[d];
        }
        map.h_output_block_offsets[block_idx] = current_offset;
        map.h_output_block_volumes[block_idx] = volume;
        map.h_output_pair_offsets[block_idx] = pair_cursor;

        // Assign all pairs in this group
        int tier_count[3] = {0, 0, 0};
        for (int64_t k = i; k < j; ++k) {
            const int64_t p_idx = entries[k].pair_idx;
            map.h_dest_offsets[p_idx] = current_offset;
            map.h_pair_output_block_idx[p_idx] = block_idx;
            map.h_pair_reuse_counts[p_idx] = reuse;
            map.h_output_pair_indices[pair_cursor++] = p_idx;
            if (work_tier) tier_count[work_tier[p_idx]]++;
        }

        // Set per-output-block tier flags
        if (work_tier) {
            uint8_t flags = 0;
            if (tier_count[0] > 0) flags |= kOBHasSmall;
            if (tier_count[1] > 0) flags |= kOBHasMedium;
            if (tier_count[2] > 0) flags |= kOBHasLarge;
            if (tier_count[0] > 1) flags |= kOBMultiSmall;
            if (tier_count[1] > 1) flags |= kOBMultiMedium;
            if (tier_count[2] > 1) flags |= kOBMultiLarge;
            map.h_ob_tier_flags[block_idx] = flags;
        }

        current_offset += volume;
        block_idx++;
        i = j;
    }
    map.h_output_pair_offsets[num_output_blocks] = pair_cursor;

    map.total_output_elements = current_offset;
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
    map.d_dest_offsets = nullptr;
    map.h_ob_tier_flags = nullptr;

    const int32_t rank = config.output_rank;
    const int64_t nob = obm.num_blocks;
    map.num_output_blocks = nob;

    // Allocate host arrays
    map.h_dest_offsets = new int64_t[num_pairs];
    map.h_pair_reuse_counts = new int32_t[num_pairs];
    map.h_pair_output_block_idx = new int64_t[num_pairs];
    map.h_output_block_coords = new int32_t[nob * rank];
    map.h_output_block_offsets = new int64_t[nob];
    map.h_output_block_volumes = new int64_t[nob];
    map.h_output_pair_offsets = new int64_t[nob + 1];
    map.h_output_pair_indices = new int64_t[num_pairs];
    if (work_tier) {
        map.h_ob_tier_flags = new uint8_t[nob];
        std::memset(map.h_ob_tier_flags, 0, static_cast<size_t>(nob) * sizeof(uint8_t));
    }

    // Copy block-level data from OutputBlockMap
    std::memcpy(map.h_output_block_coords, obm.block_coords.data(),
                static_cast<size_t>(nob * rank) * sizeof(int32_t));
    std::memcpy(map.h_output_block_offsets, obm.block_offsets.data(),
                static_cast<size_t>(nob) * sizeof(int64_t));
    std::memcpy(map.h_output_block_volumes, obm.block_volumes.data(),
                static_cast<size_t>(nob) * sizeof(int64_t));

    // Pass 1: assign each pair to its output block via hash lookup
    std::vector<int32_t> pairs_per_block(static_cast<size_t>(nob), 0);
    for (int64_t i = 0; i < num_pairs; ++i) {
        auto key = extract_output_key(pairs_host[i], A_gpu, B_gpu, config);

        auto it = obm.coord_to_idx.find(key);
        const int32_t ob = it->second;
        map.h_pair_output_block_idx[i] = ob;
        map.h_dest_offsets[i] = obm.block_offsets[static_cast<size_t>(ob)];
        pairs_per_block[static_cast<size_t>(ob)]++;
    }

    // Set reuse counts from pairs_per_block
    for (int64_t i = 0; i < num_pairs; ++i) {
        map.h_pair_reuse_counts[i] =
            pairs_per_block[static_cast<size_t>(map.h_pair_output_block_idx[i])];
    }

    // Prefix sum for pair list offsets
    int64_t pair_offset = 0;
    for (int64_t ob = 0; ob < nob; ++ob) {
        map.h_output_pair_offsets[ob] = pair_offset;
        pair_offset += pairs_per_block[static_cast<size_t>(ob)];
    }
    map.h_output_pair_offsets[nob] = pair_offset;

    // Fill pair indices grouped by output block
    std::vector<int32_t> cursor(static_cast<size_t>(nob), 0);
    for (int64_t i = 0; i < num_pairs; ++i) {
        const int64_t ob = map.h_pair_output_block_idx[i];
        const int64_t slot = map.h_output_pair_offsets[ob] + cursor[static_cast<size_t>(ob)]++;
        map.h_output_pair_indices[slot] = i;
    }

    // Tier flags
    if (work_tier) {
        for (int64_t ob = 0; ob < nob; ++ob) {
            int tier_count[3] = {0, 0, 0};
            const int64_t begin = map.h_output_pair_offsets[ob];
            const int64_t end = map.h_output_pair_offsets[ob + 1];
            for (int64_t j = begin; j < end; ++j)
                tier_count[work_tier[map.h_output_pair_indices[j]]]++;
            uint8_t flags = 0;
            if (tier_count[0] > 0) flags |= kOBHasSmall;
            if (tier_count[1] > 0) flags |= kOBHasMedium;
            if (tier_count[2] > 0) flags |= kOBHasLarge;
            if (tier_count[0] > 1) flags |= kOBMultiSmall;
            if (tier_count[1] > 1) flags |= kOBMultiMedium;
            if (tier_count[2] > 1) flags |= kOBMultiLarge;
            map.h_ob_tier_flags[static_cast<size_t>(ob)] = flags;
        }
    }

    map.total_output_elements = obm.total_elements;
    return map;
}

// Upload accumulation map to device (only d_dest_offsets needed on GPU)
inline void upload_accumulation_map(AccumulationMap& map, cudaStream_t stream = 0) {
    if (map.num_pairs > 0) {
        check_cuda(cudaMalloc(&map.d_dest_offsets,
                              map.num_pairs * sizeof(int64_t)),
                   "malloc d_dest_offsets");
        check_cuda(cudaMemcpyAsync(map.d_dest_offsets, map.h_dest_offsets,
                                    map.num_pairs * sizeof(int64_t),
                                    cudaMemcpyHostToDevice, stream),
                   "memcpy d_dest_offsets");
    }
}

// Free all accumulation map memory
inline void free_accumulation_map(AccumulationMap& map) {
    delete[] map.h_dest_offsets;
    delete[] map.h_pair_reuse_counts;
    delete[] map.h_pair_output_block_idx;
    delete[] map.h_output_block_coords;
    delete[] map.h_output_block_offsets;
    delete[] map.h_output_block_volumes;
    delete[] map.h_output_pair_offsets;
    delete[] map.h_output_pair_indices;
    delete[] map.h_ob_tier_flags;
    map.h_dest_offsets = nullptr;
    map.h_pair_reuse_counts = nullptr;
    map.h_pair_output_block_idx = nullptr;
    map.h_output_block_coords = nullptr;
    map.h_output_block_offsets = nullptr;
    map.h_output_block_volumes = nullptr;
    map.h_output_pair_offsets = nullptr;
    map.h_output_pair_indices = nullptr;
    map.h_ob_tier_flags = nullptr;

    if (map.d_dest_offsets) {
        cudaFree(map.d_dest_offsets);
        map.d_dest_offsets = nullptr;
    }
}

inline void free_reduction_plan(ReductionPlan& plan) {
    delete[] plan.h_direct_write;
    delete[] plan.h_reduce_dest_offsets;
    delete[] plan.h_reduce_volumes;
    delete[] plan.h_reduce_src_offsets;
    delete[] plan.h_reduce_src_ptrs;
    plan.h_direct_write = nullptr;
    plan.h_reduce_dest_offsets = nullptr;
    plan.h_reduce_volumes = nullptr;
    plan.h_reduce_src_offsets = nullptr;
    plan.h_reduce_src_ptrs = nullptr;

    if (plan.d_reduce_dest_offsets) { cudaFree(plan.d_reduce_dest_offsets); }
    if (plan.d_reduce_volumes) { cudaFree(plan.d_reduce_volumes); }
    if (plan.d_reduce_src_offsets) { cudaFree(plan.d_reduce_src_offsets); }
    if (plan.d_reduce_src_ptrs) { cudaFree(plan.d_reduce_src_ptrs); }
    plan.d_reduce_dest_offsets = nullptr;
    plan.d_reduce_volumes = nullptr;
    plan.d_reduce_src_offsets = nullptr;
    plan.d_reduce_src_ptrs = nullptr;
    plan.num_pairs = 0;
    plan.num_reduce_groups = 0;
}

// Build reduction plan for the CUTLASS tier (only tier using temp buffer):
// - `h_direct_write[p] == 1` means pair p can write directly to final output
// - output blocks with >1 temp-buffer pairs become no-atomic reduction groups
// pair_tier: 0 = warp/volume (write to d_output, no temp buffer),
//            2 = CUTLASS (temp buffer, eligible for direct-write singleton).
// ob_tier_flags: per-output-block tier bitfield (h_ob_tier_flags from AccumulationMap).
//   CUTLASS pairs on mixed blocks (kOBHasLarge & kOBHasMedium) must NOT direct-write
//   because volume pairs write to d_output concurrently.  May be nullptr (no mixed blocks).
inline ReductionPlan build_reduction_plan(
    const AccumulationMap& map,
    const std::vector<uint8_t>& pair_tier,
    const BlockPairInfo* pairs_host,
    const uint8_t* ob_tier_flags = nullptr)
{
    ReductionPlan plan;
    plan.num_pairs = map.num_pairs;

    if (map.num_pairs > 0) {
        plan.h_direct_write = new uint8_t[map.num_pairs];
        std::memset(plan.h_direct_write, 0, map.num_pairs * sizeof(uint8_t));
    }

    if (map.num_pairs == 0 || map.num_output_blocks == 0 ||
        pair_tier.size() != static_cast<size_t>(map.num_pairs)) {
        return plan;
    }

    // Count temp-buffer pairs per output block.
    // Volume pairs (tier 0/1) write to d_output, not temp buffer.
    // On mixed blocks (both volume and CUTLASS), CUTLASS goes to temp buffer
    // and reduction merges temp → d_output after all tiers complete.
    std::vector<int64_t> tb_count_by_block(static_cast<size_t>(map.num_output_blocks), 0);
    for (int64_t p = 0; p < map.num_pairs; ++p) {
        if (pair_tier[p] != 0) {
            const int64_t out_block = map.h_pair_output_block_idx[p];
            tb_count_by_block[static_cast<size_t>(out_block)]++;
        }
    }

    // Pass 1: identify direct-write singletons.
    // Eligible when exactly 1 temp-buffer pair for the output block (any tier != 0)
    // AND the block is not mixed (no volume pairs writing to d_output concurrently).
    for (int64_t out_block = 0; out_block < map.num_output_blocks; ++out_block) {
        if (tb_count_by_block[static_cast<size_t>(out_block)] != 1) continue;
        if (ob_tier_flags &&
            (ob_tier_flags[out_block] & (kOBHasLarge | kOBHasMedium)) ==
            (kOBHasLarge | kOBHasMedium)) continue;
        const int64_t begin = map.h_output_pair_offsets[out_block];
        const int64_t end = map.h_output_pair_offsets[out_block + 1];
        for (int64_t i = begin; i < end; ++i) {
            const int64_t pair_idx = map.h_output_pair_indices[i];
            if (pair_tier[pair_idx] != 0) {
                plan.h_direct_write[pair_idx] = 1;
                break;
            }
        }
    }

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
