#pragma once

// Host-side hash join for block pair matching.
// Uses gtl::flat_hash_map (Swiss table) with fixed-size keys templated on
// key length (num_contracted + num_batch) for compile-time unrolling.
//
// Separated from preprocessing_kernel.cuh to avoid gtl/CUTLASS symbol
// collision (both define umul128 in the global namespace). Only include
// this header in TUs that need find_matching_pairs.

#include "coord_key.hpp"

#include <vector>

#include <gtl/phmap.hpp>

namespace bsptc {

// Aliases for backward compatibility — pair matching uses CoordKey from coord_key.hpp.
template <int N> using MatchKey = CoordKey<N>;
template <int N> using MatchKeyHash = CoordKeyHash<N>;

// Extract fixed-size match key from block coordinates.
template <int N>
inline MatchKey<N> extract_match_key_fixed(
    const int32_t* block_coords,
    dim_t rank,
    int64_t blk_idx,
    const int32_t* contracted_pos,
    dim_t num_contracted,
    const int32_t* batch_pos,
    dim_t num_batch)
{
    return extract_coord_key<N>(block_coords, rank, blk_idx,
                                contracted_pos, num_contracted,
                                batch_pos, num_batch);
}

// Hash join implementation templated on key length N.
// Uses gtl::flat_hash_map (Swiss table) with fixed-size keys.
template <int N, typename T>
std::vector<BlockPairInfo> find_matching_pairs_impl(
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const ContractionConfig& config)
{
    using Key = MatchKey<N>;
    using Map = gtl::flat_hash_map<Key, std::vector<int32_t>, MatchKeyHash<N>>;

    // Build phase: group A block indices by contracted+batch key
    Map a_map;
    a_map.reserve(A_gpu.num_blocks);

    for (int32_t ai = 0; ai < A_gpu.num_blocks; ++ai) {
        auto key = extract_match_key_fixed<N>(
            A_gpu.block_coords.data(), A_gpu.rank, ai,
            config.contracted_pos_A, config.num_contracted,
            config.batch_pos_A, config.num_batch);
        a_map[key].push_back(ai);
    }

    // Probe phase: look up each B block's key, emit pairs
    std::vector<BlockPairInfo> pairs;
    for (int32_t bi = 0; bi < B_gpu.num_blocks; ++bi) {
        auto key = extract_match_key_fixed<N>(
            B_gpu.block_coords.data(), B_gpu.rank, bi,
            config.contracted_pos_B, config.num_contracted,
            config.batch_pos_B, config.num_batch);

        auto it = a_map.find(key);
        if (it == a_map.end()) continue;

        for (int32_t ai : it->second) {
            BlockPairInfo info;
            info.block_idx_A = ai;
            info.block_idx_B = bi;
            info.output_offset = 0;  // computed by caller via prefix sum
            info.output_volume = compute_output_volume_host(
                A_gpu.block_dims.data(), B_gpu.block_dims.data(),
                A_gpu.rank, B_gpu.rank,
                ai, bi, config);
            pairs.push_back(info);
        }
    }

    return pairs;
}

// Dynamic-N aliases
using MatchKeyDynamic = CoordKeyDynamic;
using MatchKeyDynamicHash = CoordKeyDynamicHash;

inline MatchKeyDynamic extract_match_key_dynamic(
    const int32_t* block_coords,
    dim_t rank,
    int64_t blk_idx,
    const int32_t* contracted_pos,
    dim_t num_contracted,
    const int32_t* batch_pos,
    dim_t num_batch)
{
    return extract_coord_key_dynamic(block_coords, rank, blk_idx,
                                     contracted_pos, num_contracted,
                                     batch_pos, num_batch);
}

template <typename T>
std::vector<BlockPairInfo> find_matching_pairs_dynamic(
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const ContractionConfig& config)
{
    using Map = gtl::flat_hash_map<MatchKeyDynamic, std::vector<int32_t>,
                                    MatchKeyDynamicHash>;

    Map a_map;
    a_map.reserve(A_gpu.num_blocks);

    for (int32_t ai = 0; ai < A_gpu.num_blocks; ++ai) {
        auto key = extract_match_key_dynamic(
            A_gpu.block_coords.data(), A_gpu.rank, ai,
            config.contracted_pos_A, config.num_contracted,
            config.batch_pos_A, config.num_batch);
        a_map[key].push_back(ai);
    }

    std::vector<BlockPairInfo> pairs;
    for (int32_t bi = 0; bi < B_gpu.num_blocks; ++bi) {
        auto key = extract_match_key_dynamic(
            B_gpu.block_coords.data(), B_gpu.rank, bi,
            config.contracted_pos_B, config.num_contracted,
            config.batch_pos_B, config.num_batch);

        auto it = a_map.find(key);
        if (it == a_map.end()) continue;

        for (int32_t ai : it->second) {
            BlockPairInfo info;
            info.block_idx_A = ai;
            info.block_idx_B = bi;
            info.output_offset = 0;
            info.output_volume = compute_output_volume_host(
                A_gpu.block_dims.data(), B_gpu.block_dims.data(),
                A_gpu.rank, B_gpu.rank,
                ai, bi, config);
            pairs.push_back(info);
        }
    }

    return pairs;
}

// Host-side hash join: find all matching block pairs.
// Dispatches to a template specialization based on key length for
// compile-time unrolling of the hash and comparison.
template <typename T>
std::vector<BlockPairInfo> find_matching_pairs(
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const ContractionConfig& config)
{
    int key_len = config.num_contracted + config.num_batch;
    switch (key_len) {
        case 1:  return find_matching_pairs_impl<1>(A_gpu, B_gpu, config);
        case 2:  return find_matching_pairs_impl<2>(A_gpu, B_gpu, config);
        case 3:  return find_matching_pairs_impl<3>(A_gpu, B_gpu, config);
        case 4:  return find_matching_pairs_impl<4>(A_gpu, B_gpu, config);
        case 5:  return find_matching_pairs_impl<5>(A_gpu, B_gpu, config);
        case 6:  return find_matching_pairs_impl<6>(A_gpu, B_gpu, config);
        case 7:  return find_matching_pairs_impl<7>(A_gpu, B_gpu, config);
        case 8:  return find_matching_pairs_impl<8>(A_gpu, B_gpu, config);
        default: return find_matching_pairs_dynamic(A_gpu, B_gpu, config);
    }
}

} // namespace bsptc
