#pragma once

// Thin header for hash-table-based dedup functions.
// Implementation lives in src/hash_tables.cpp (isolated from CUTLASS headers
// to avoid gtl/CUTLASS umul128 symbol collision).
//
// No gtl, CUDA, or domain-type includes — depends only on types.hpp + stdlib.

#include "../types.hpp"
#include <cstdint>

namespace bsptc {

struct BlockShapeMap;  // defined in block_shape_map.hpp

// Build per-tensor block shape map via gtl::flat_hash_map.
// block_dims: [num_blocks * rank] flattened, row-major.
BlockShapeMap build_block_shape_map(const dim_t* block_dims, int num_blocks, dim_t rank);

// Deduplicate uint64_t keys (e.g. pair shape keys from make_pair_shape_key).
// Writes per-item unique index to indices_out[n].
// Writes input position of first occurrence of each unique key to first_occurrence_out[].
// Both arrays must be pre-allocated by the caller (n elements each, worst-case).
// Returns number of unique keys.
int pair_key_dedup(const uint64_t* keys, int n,
                   uint32_t* indices_out,
                   int* first_occurrence_out);

// Group rows of a flat [n * rank] int32 array by row value.
// Group ids are assigned in first-occurrence order; group_of[i] receives the
// id of row i (caller allocates n entries). Returns the number of distinct
// groups. Used by the accumulation-map discovery path (O(n), replaces sort).
int64_t group_coord_rows(const int32_t* flat_coords, int64_t n, int32_t rank,
                         int32_t* group_of);

} // namespace bsptc
