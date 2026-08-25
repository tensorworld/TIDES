#pragma once

// build_hypertile_index: constructs a HypertileIndex from pipeline types.
// Separated from nd_hypertile.cuh because it depends on GPUSparseInput,
// GemmLayoutPlan, and BlockShapeMap — pipeline-level types that the kernel
// header should not require.

#include "nd_hypertile.cuh"
#include "../../gpu_sparse_input.hpp"
#include "../../gemm_layout_check.hpp"
#include "../../block_shape_map.hpp"

namespace bsptc {

// Build hypertile index from an explicit permutation over unique shapes.
// full_perm[src_dim] = destination position of src_dim (row-major both sides).
// Shared core for the operand pre-GEMM transposes and the post-contraction
// output permute.
inline HypertileIndex build_hypertile_index_perm(
    const BlockShapeMap& shape_map,
    dim_t rank,
    const int32_t* full_perm,
    int tile_vol_limit = 1024)
{
    HypertileIndex index;
    if (shape_map.num_unique == 0) return index;

    std::vector<int32_t> inv_perm(rank);
    for (int32_t sd = 0; sd < rank; ++sd)
        inv_perm[full_perm[sd]] = sd;

    const int num_shapes = shape_map.num_unique;
    index.per_shape.resize(num_shapes);

    // WR → slice index mapping (lazily assigned)
    int wr_to_slice[HT_MAX_RANK + 1];
    std::memset(wr_to_slice, -1, sizeof(wr_to_slice));

    // Single pass: build shape → determine WR → pack into correct slice
    for (int s = 0; s < num_shapes; ++s) {
        const dim_t* dims = shape_map.shape_dims(static_cast<uint32_t>(s));
        auto r = build_task_for_shape(dims, rank, rank,
                                       full_perm, inv_perm.data(),
                                       tile_vol_limit);
        if (r.rank_overflow) {
            index.rank_overflow = true;
            index.valid = false;
            return index;  // caller must fail; skipping would drop the transpose
        }
        if (r.skip) {
            index.per_shape[s] = {0, 0, 0, 0, true};
            continue;
        }

        const int wr = r.shape.WR;

        // Find or create slice for this WR
        if (wr_to_slice[wr] < 0) {
            wr_to_slice[wr] = index.num_slices;
            index.slices[index.num_slices].rank = wr;
            index.num_slices++;
        }
        int si_idx = wr_to_slice[wr];
        auto& si = index.slices[si_idx];

        // Pack shape into slice byte buffer (switch on WR for sizeof)
        uint32_t shape_idx = static_cast<uint32_t>(si.num_shapes);
        si.h_num_tiles.push_back(r.shape.num_tiles);

        auto append_packed = [&](auto rank_tag) {
            constexpr int R = decltype(rank_tag)::value;
            auto packed = pack_shape<R>(r.shape);
            size_t offset = si.h_shape_bytes.size();
            si.h_shape_bytes.resize(offset + sizeof(packed));
            std::memcpy(si.h_shape_bytes.data() + offset, &packed, sizeof(packed));
        };
        switch (wr) {
            case 2: append_packed(std::integral_constant<int,2>{}); break;
            case 3: append_packed(std::integral_constant<int,3>{}); break;
            case 4: append_packed(std::integral_constant<int,4>{}); break;
            case 5: append_packed(std::integral_constant<int,5>{}); break;
            case 6: append_packed(std::integral_constant<int,6>{}); break;
            case 7: append_packed(std::integral_constant<int,7>{}); break;
            case 8: append_packed(std::integral_constant<int,8>{}); break;
            default: break;
        }
        si.num_shapes++;

        auto& entry = index.per_shape[s];
        entry.rank = static_cast<uint8_t>(wr);
        entry.slice_idx = static_cast<uint8_t>(si_idx);
        entry.shape_idx = shape_idx;
        entry.num_tiles = r.shape.num_tiles;
        entry.trivial = false;

        // Update diagnostics
        index.max_squashed_rank = std::max(index.max_squashed_rank,
            static_cast<int32_t>(r.shape.squashed_rank));
        index.max_tile_volume = std::max(index.max_tile_volume, r.shape.tile_volume);
        index.min_rr = std::min(index.min_rr, static_cast<int32_t>(r.shape.read.RR));
        index.min_wr = std::min(index.min_wr, static_cast<int32_t>(r.shape.WR));
    }

    // Finalize shape_bytes per slice
    for (int i = 0; i < index.num_slices; ++i)
        index.slices[i].shape_bytes = index.slices[i].h_shape_bytes.size();

    index.valid = true;
    return index;
}

// Build hypertile index for an operand's pre-GEMM transpose. One index per
// operand, built once during pipeline_prepare.
template <typename T>
HypertileIndex build_hypertile_index(
    const GPUSparseInput<T>& gpu,
    const typename GemmLayoutPlan::OperandPlan& op_plan,
    const BlockShapeMap& shape_map,
    int tile_vol_limit = 1024)
{
    if (gpu.num_blocks == 0) return HypertileIndex{};

    const dim_t rank = gpu.rank;
    const dim_t head_rank = rank - op_plan.unmoved_tail_dims;

    // Build full-rank perm
    std::vector<int32_t> full_perm(rank);
    for (int32_t d = 0; d < head_rank; ++d) full_perm[d] = op_plan.perm[d];
    for (int32_t d = head_rank; d < rank; ++d) full_perm[d] = d;

    return build_hypertile_index_perm(shape_map, rank, full_perm.data(),
                                      tile_vol_limit);
}

} // namespace bsptc
