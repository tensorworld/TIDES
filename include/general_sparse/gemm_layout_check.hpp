#pragma once

#include "preprocessing_kernel.cuh"

#include <stdexcept>
#include <vector>

// GEMM layout compatibility check for CUTLASS dispatch.
// CUTLASS interprets block data as row-major GEMM:
//   A: [batch, M, K] → modes must appear in order [batch..., free_A..., contracted...]
//   B: [batch, K, N] → modes must appear in order [batch..., contracted..., free_B...]
// If a tensor's mode ordering violates this grouping (e.g., contracted dims
// interleaved with free dims), lda=K is wrong and CUTLASS produces garbage.
namespace bsptc {

enum class DimRole { Batch, Free, Contracted };
enum class GemmOperandLayout { Normal, Transposed, Interleaved };
enum class GemmMode { NN, TN, NT, TT };
enum class TransposeDispatchKind { None, GenericPermute };

inline bool gemm_mode_a_transposed(GemmMode mode) {
    return mode == GemmMode::TN || mode == GemmMode::TT;
}

inline bool gemm_mode_b_transposed(GemmMode mode) {
    return mode == GemmMode::NT || mode == GemmMode::TT;
}

inline const char* gemm_mode_to_string(GemmMode mode) {
    switch (mode) {
        case GemmMode::NN: return "NN";
        case GemmMode::TN: return "TN";
        case GemmMode::NT: return "NT";
        case GemmMode::TT: return "TT";
    }
    return "NN";
}

// Role-parameterized view of one operand's dim-group positions. The two
// operands differ only in which config arrays hold their positions and in
// the "normal" group order: A-normal is [batch, free, contracted], B-normal
// is [batch, contracted, free] — each role's normal is the other's
// transposed, so a single implementation serves both.
struct OperandDimGroups {
    const int32_t* batch_pos;
    const int32_t* free_pos;
    const int32_t* contracted_pos;
    int32_t num_batch;
    int32_t num_free;
    int32_t num_contracted;
    dim_t rank;
    bool free_first;  // true for A ([B,F,C] normal), false for B ([B,C,F] normal)
};

inline OperandDimGroups operand_dim_groups(const ContractionConfig& config,
                                           dim_t rank, bool is_A) {
    if (is_A) {
        return {config.batch_pos_A, config.free_pos_A, config.contracted_pos_A,
                config.num_batch, config.num_free_A, config.num_contracted,
                rank, true};
    }
    return {config.batch_pos_B, config.free_pos_B, config.contracted_pos_B,
            config.num_batch, config.num_free_B, config.num_contracted,
            rank, false};
}

// Check that the operand's dims appear grouped in the required order:
// batch first, then free/contracted in the order given by free_first
// (flipped when transposed).
inline bool check_group_order(const OperandDimGroups& od,
                              const DimRole roles[MAX_RANK],
                              const bool assigned[MAX_RANK],
                              bool transposed) {
    const int free_group = (od.free_first != transposed) ? 1 : 2;
    int max_group = -1;
    for (int32_t p = 0; p < od.rank; ++p) {
        if (!assigned[p]) continue;
        int group = 0;
        if (roles[p] == DimRole::Free)
            group = free_group;
        else if (roles[p] == DimRole::Contracted)
            group = 3 - free_group;
        if (group < max_group) return false;
        max_group = group;
    }
    return true;
}

// Check that positions for a dim group are in config-index order (monotonically
// increasing).  When they're not, linearized iteration picks wrong slices.
inline bool positions_in_order(const int32_t* pos, int32_t count) {
    for (int32_t i = 1; i < count; ++i)
        if (pos[i] < pos[i - 1]) return false;
    return true;
}

inline GemmOperandLayout classify_gemm_layout(const OperandDimGroups& od) {
    DimRole roles[MAX_RANK];
    bool assigned[MAX_RANK] = {};

    for (int32_t i = 0; i < od.num_batch; ++i) {
        roles[od.batch_pos[i]] = DimRole::Batch;
        assigned[od.batch_pos[i]] = true;
    }
    for (int32_t i = 0; i < od.num_free; ++i) {
        roles[od.free_pos[i]] = DimRole::Free;
        assigned[od.free_pos[i]] = true;
    }
    for (int32_t i = 0; i < od.num_contracted; ++i) {
        roles[od.contracted_pos[i]] = DimRole::Contracted;
        assigned[od.contracted_pos[i]] = true;
    }

    // Batch dims must be in config-index order for linearized batch iteration
    if (!positions_in_order(od.batch_pos, od.num_batch))
        return GemmOperandLayout::Interleaved;

    // Contracted dims must be in config-index order so flattened K matches
    // between operands (permutation functions produce config-index order)
    if (!positions_in_order(od.contracted_pos, od.num_contracted))
        return GemmOperandLayout::Interleaved;

    if (check_group_order(od, roles, assigned, false))
        return GemmOperandLayout::Normal;
    if (check_group_order(od, roles, assigned, true))
        return GemmOperandLayout::Transposed;
    return GemmOperandLayout::Interleaved;
}

inline GemmMode gemm_mode_from_layouts(GemmOperandLayout a_layout, GemmOperandLayout b_layout) {
    if (a_layout == GemmOperandLayout::Interleaved ||
        b_layout == GemmOperandLayout::Interleaved) {
        throw std::runtime_error("Cannot compute GEMM mode from interleaved layout");
    }

    bool a_t = (a_layout == GemmOperandLayout::Transposed);
    bool b_t = (b_layout == GemmOperandLayout::Transposed);
    if (!a_t && !b_t) return GemmMode::NN;
    if (a_t && !b_t) return GemmMode::TN;
    if (!a_t && b_t) return GemmMode::NT;
    return GemmMode::TT;
}

struct GemmLayoutPlan {
    GemmOperandLayout layout_A = GemmOperandLayout::Normal;
    GemmOperandLayout layout_B = GemmOperandLayout::Normal;
    struct OperandPlan {
        GemmOperandLayout original_layout = GemmOperandLayout::Normal;
        GemmOperandLayout target_layout = GemmOperandLayout::Normal;
        bool transpose = false;
        TransposeDispatchKind dispatch_kind = TransposeDispatchKind::None;
        dim_t unmoved_tail_dims = 0;
        std::vector<int32_t> perm;
    } plan_A, plan_B;
    bool transpose_A = false;
    bool transpose_B = false;
    GemmMode mode_after_transpose = GemmMode::NN;
};

inline dim_t count_moved_dims(const std::vector<int32_t>& perm) {
    dim_t moved = 0;
    for (int32_t i = 0; i < static_cast<int32_t>(perm.size()); ++i) {
        if (perm[i] != i) moved++;
    }
    return moved;
}

inline dim_t count_unmoved_tail_dims(const std::vector<int32_t>& perm) {
    dim_t tail = 0;
    for (int32_t i = static_cast<int32_t>(perm.size()) - 1; i >= 0; --i) {
        if (perm[i] == i) {
            tail++;
        } else {
            break;
        }
    }
    return tail;
}

// Compute permutation to reorder the operand's dims to one of the
// GEMM-compatible groupings.  perm[src_dim] = dst_dim in the selected ordering.
inline std::vector<int32_t> compute_gemm_permutation_for_layout(
    const OperandDimGroups& od, GemmOperandLayout target_layout) {
    if (target_layout == GemmOperandLayout::Interleaved) {
        throw std::runtime_error("target layout cannot be interleaved");
    }

    std::vector<int32_t> perm(od.rank);
    int32_t dst = 0;
    for (int32_t i = 0; i < od.num_batch; ++i)
        perm[od.batch_pos[i]] = dst++;
    const bool free_next = (target_layout == GemmOperandLayout::Normal) == od.free_first;
    if (free_next) {
        for (int32_t i = 0; i < od.num_free; ++i)
            perm[od.free_pos[i]] = dst++;
        for (int32_t i = 0; i < od.num_contracted; ++i)
            perm[od.contracted_pos[i]] = dst++;
    } else {
        for (int32_t i = 0; i < od.num_contracted; ++i)
            perm[od.contracted_pos[i]] = dst++;
        for (int32_t i = 0; i < od.num_free; ++i)
            perm[od.free_pos[i]] = dst++;
    }
    return perm;
}

struct PermutationCandidate {
    GemmOperandLayout target_layout = GemmOperandLayout::Normal;
    std::vector<int32_t> perm;
    dim_t moved_dims = 0;
    dim_t unmoved_tail_dims = 0;
};

inline bool is_better_candidate(const PermutationCandidate& lhs,
                                const PermutationCandidate& rhs) {
    if (lhs.unmoved_tail_dims != rhs.unmoved_tail_dims) {
        return lhs.unmoved_tail_dims > rhs.unmoved_tail_dims;
    }
    if (lhs.moved_dims != rhs.moved_dims) {
        return lhs.moved_dims < rhs.moved_dims;
    }
    // Deterministic tie-breaker: keep current normal-first behavior.
    return lhs.target_layout == GemmOperandLayout::Normal &&
           rhs.target_layout != GemmOperandLayout::Normal;
}

inline PermutationCandidate make_candidate(const OperandDimGroups& od,
                                           GemmOperandLayout target_layout) {
    PermutationCandidate candidate;
    candidate.target_layout = target_layout;
    candidate.perm = compute_gemm_permutation_for_layout(od, target_layout);
    candidate.moved_dims = count_moved_dims(candidate.perm);
    candidate.unmoved_tail_dims = count_unmoved_tail_dims(candidate.perm);
    return candidate;
}

inline GemmLayoutPlan::OperandPlan plan_operand(const OperandDimGroups& od,
                                                GemmOperandLayout current_layout) {
    GemmLayoutPlan::OperandPlan out;
    out.original_layout = current_layout;
    out.target_layout = current_layout;
    out.transpose = (current_layout == GemmOperandLayout::Interleaved);
    if (!out.transpose) {
        return out;
    }

    PermutationCandidate normal = make_candidate(od, GemmOperandLayout::Normal);
    PermutationCandidate transposed = make_candidate(od, GemmOperandLayout::Transposed);
    const PermutationCandidate& best =
        is_better_candidate(transposed, normal) ? transposed : normal;

    out.target_layout = best.target_layout;
    out.perm = best.perm;
    out.unmoved_tail_dims = best.unmoved_tail_dims;
    out.dispatch_kind = TransposeDispatchKind::GenericPermute;
    return out;
}

inline GemmLayoutPlan plan_gemm_layout(const ContractionConfig& config,
                                       dim_t A_rank,
                                       dim_t B_rank) {
    const OperandDimGroups od_A = operand_dim_groups(config, A_rank, /*is_A=*/true);
    const OperandDimGroups od_B = operand_dim_groups(config, B_rank, /*is_A=*/false);

    GemmLayoutPlan plan;
    plan.layout_A = classify_gemm_layout(od_A);
    plan.layout_B = classify_gemm_layout(od_B);

    plan.plan_A = plan_operand(od_A, plan.layout_A);
    plan.plan_B = plan_operand(od_B, plan.layout_B);
    plan.transpose_A = plan.plan_A.transpose;
    plan.transpose_B = plan.plan_B.transpose;
    plan.mode_after_transpose =
        gemm_mode_from_layouts(plan.plan_A.target_layout, plan.plan_B.target_layout);
    return plan;
}

} // namespace bsptc
