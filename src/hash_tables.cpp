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

namespace {

// Non-owning row reference with precomputed hash. Rows live in the caller's
// flat_coords array, which outlives the table — no per-key copies.
struct RowRef {
    const int32_t* p;
    uint64_t hash;
};

struct RowRefHash {
    size_t operator()(const RowRef& r) const noexcept {
        return static_cast<size_t>(r.hash);
    }
};

struct RowRefEq {
    size_t row_bytes;
    bool operator()(const RowRef& a, const RowRef& b) const noexcept {
        return a.hash == b.hash && std::memcmp(a.p, b.p, row_bytes) == 0;
    }
};

} // namespace

int64_t group_coord_rows(const int32_t* flat_coords, int64_t n, int32_t rank,
                         int32_t* group_of) {
    const size_t row_bytes = static_cast<size_t>(rank) * sizeof(int32_t);
    gtl::flat_hash_map<RowRef, int32_t, RowRefHash, RowRefEq>
        table(0, RowRefHash{}, RowRefEq{row_bytes});
    table.reserve(static_cast<size_t>(n));

    int32_t num_groups = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int32_t* p = flat_coords + i * rank;
        uint64_t h = 0xcbf29ce484222325ULL;
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(p);
        for (size_t b = 0; b < row_bytes; ++b) {
            h ^= bytes[b];
            h *= 0x100000001b3ULL;
        }
        auto [it, inserted] = table.try_emplace(RowRef{p, h}, num_groups);
        if (inserted) num_groups++;
        group_of[i] = it->second;
    }
    return num_groups;
}

} // namespace bsptc
