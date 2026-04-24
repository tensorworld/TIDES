#pragma once

// Fixed-size and dynamic coordinate keys for hash-map lookups.
// Used by pair matching (contracted+batch keys) and output block mapping
// (batch+free_A+free_B keys). No gtl dependency — safe in CUTLASS TUs.

#include "preprocessing_kernel.cuh"  // MAX_RANK, dim_t

#include <cstdint>
#include <cstring>
#include <functional>

namespace bsptc {

// Fixed-size key templated on N = number of coordinate dimensions.
template <int N>
struct CoordKey {
    int32_t coords[N];

    bool operator==(const CoordKey& o) const {
        return std::memcmp(coords, o.coords, N * sizeof(int32_t)) == 0;
    }
};

template <int N>
struct CoordKeyHash {
    size_t operator()(const CoordKey<N>& k) const {
        size_t h = 0;
        for (int i = 0; i < N; ++i)
            h ^= std::hash<int32_t>{}(k.coords[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// Runtime-length key (up to MAX_RANK dimensions).
struct CoordKeyDynamic {
    int32_t coords[MAX_RANK] = {};
    int32_t len = 0;

    bool operator==(const CoordKeyDynamic& o) const {
        return len == o.len &&
               std::memcmp(coords, o.coords, len * sizeof(int32_t)) == 0;
    }
};

struct CoordKeyDynamicHash {
    size_t operator()(const CoordKeyDynamic& k) const {
        size_t h = 0;
        for (int i = 0; i < k.len; ++i)
            h ^= std::hash<int32_t>{}(k.coords[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// Extract a fixed-size key from flat block_coords.
template <int N>
inline CoordKey<N> extract_coord_key(
    const int32_t* block_coords,
    dim_t rank,
    int64_t blk_idx,
    const int32_t* pos_list_1,
    dim_t count_1,
    const int32_t* pos_list_2 = nullptr,
    dim_t count_2 = 0)
{
    CoordKey<N> key{};
    int idx = 0;
    for (int32_t i = 0; i < count_1; ++i)
        key.coords[idx++] = block_coords[blk_idx * rank + pos_list_1[i]];
    for (int32_t i = 0; i < count_2; ++i)
        key.coords[idx++] = block_coords[blk_idx * rank + pos_list_2[i]];
    return key;
}

// Extract a dynamic-length key from flat block_coords.
inline CoordKeyDynamic extract_coord_key_dynamic(
    const int32_t* block_coords,
    dim_t rank,
    int64_t blk_idx,
    const int32_t* pos_list_1,
    dim_t count_1,
    const int32_t* pos_list_2 = nullptr,
    dim_t count_2 = 0)
{
    CoordKeyDynamic key;
    key.len = count_1 + count_2;
    int idx = 0;
    for (int32_t i = 0; i < count_1; ++i)
        key.coords[idx++] = block_coords[blk_idx * rank + pos_list_1[i]];
    for (int32_t i = 0; i < count_2; ++i)
        key.coords[idx++] = block_coords[blk_idx * rank + pos_list_2[i]];
    return key;
}

} // namespace bsptc
