// Hash-table-based dedup functions using gtl::flat_hash_map (Swiss table).
// Isolated from CUTLASS headers to avoid umul128 symbol collision.
// Pure C++ — no CUDA device code.

#include "general_sparse/hash_tables.hpp"
#include "general_sparse/block_shape_map.hpp"

#include <gtl/phmap.hpp>
#include <cstring>

namespace bsptc {

// Fixed-size key for dim arrays: zero-padded.
// Trivially copyable — no heap allocation, stored inline in flat_hash_map.
static constexpr int DIM_KEY_MAX = 16;  // covers MAX_RANK (8) with headroom
struct DimKey {
    dim_t dims[DIM_KEY_MAX] = {};

    bool operator==(const DimKey& o) const {
        return std::memcmp(dims, o.dims, sizeof(dims)) == 0;
    }
};

struct DimKeyHash {
    size_t dim_bytes;  // rank * sizeof(dim_t), set once per call

    size_t operator()(const DimKey& k) const noexcept {
        // FNV-1a over the meaningful prefix only (rest is zero-padded,
        // so equal keys always have equal prefix → consistent hash).
        uint64_t h = 0xcbf29ce484222325ULL;
        const uint8_t* p = reinterpret_cast<const uint8_t*>(k.dims);
        for (size_t i = 0; i < dim_bytes; ++i) {
            h ^= p[i];
            h *= 0x100000001b3ULL;
        }
        return static_cast<size_t>(h);
    }
};

BlockShapeMap build_block_shape_map(const dim_t* block_dims, int num_blocks, dim_t rank) {
    BlockShapeMap map;
    map.rank = rank;
    if (num_blocks == 0) return map;
    map.block_to_shape.resize(num_blocks);

    const size_t dim_bytes = static_cast<size_t>(rank) * sizeof(dim_t);

    // Fixed-size keys: zero-padded dim array, no heap allocation per probe.
    gtl::flat_hash_map<DimKey, uint32_t, DimKeyHash> table(0, DimKeyHash{dim_bytes});
    table.reserve(num_blocks);

    for (int b = 0; b < num_blocks; ++b) {
        DimKey key{};
        std::memcpy(key.dims, block_dims + b * rank, dim_bytes);
        auto [it, inserted] = table.try_emplace(key, static_cast<uint32_t>(map.num_unique));
        if (inserted) {
            map.unique_dims.insert(map.unique_dims.end(),
                block_dims + b * rank, block_dims + b * rank + rank);
            map.num_unique++;
        }
        map.block_to_shape[b] = it->second;
    }
    return map;
}

int pair_key_dedup(const uint64_t* keys, int n,
                   uint32_t* indices_out,
                   int* first_occurrence_out) {
    gtl::flat_hash_map<uint64_t, uint32_t> table;
    table.reserve(n);
    int num_unique = 0;
    for (int i = 0; i < n; ++i) {
        auto [it, inserted] = table.try_emplace(keys[i], static_cast<uint32_t>(num_unique));
        if (inserted) {
            first_occurrence_out[num_unique] = i;
            num_unique++;
        }
        indices_out[i] = it->second;
    }
    return num_unique;
}

} // namespace bsptc
