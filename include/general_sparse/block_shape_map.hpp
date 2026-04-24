#pragma once

#include "../types.hpp"
#include <vector>
#include <cstdint>

namespace bsptc {

// Per-tensor block shape deduplication.
// Groups blocks with identical dims into shape classes, indexed by uint32_t.
struct BlockShapeMap {
    std::vector<dim_t> unique_dims;       // [num_unique * rank] flattened
    std::vector<uint32_t> block_to_shape; // [num_blocks] → shape index
    int num_unique = 0;
    dim_t rank = 0;

    const dim_t* shape_dims(uint32_t shape_id) const {
        return &unique_dims[shape_id * rank];
    }
};

// Build a BlockShapeMap by deduplicating block_dims via gtl::flat_hash_map.
// block_dims: [num_blocks * rank] flattened, row-major.
// Implemented in src/hash_tables.cpp (isolated from CUTLASS headers).
BlockShapeMap build_block_shape_map(const dim_t* block_dims, int num_blocks, dim_t rank);

// Pair shape key: encodes (A_shape_id, B_shape_id) into a single uint64_t.
inline uint64_t make_pair_shape_key(uint32_t a_shape, uint32_t b_shape) {
    return (static_cast<uint64_t>(a_shape) << 32) | b_shape;
}

} // namespace bsptc
