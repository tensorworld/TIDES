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

inline bool check_group_order_A(const DimRole roles[MAX_RANK],
                                const bool assigned[MAX_RANK],
                                dim_t A_rank,
                                bool transposed) {
    // A normal group order: [Batch, Free, Contracted]
    // A transposed group order: [Batch, Contracted, Free]
    int max_group = -1;
    for (int32_t p = 0; p < A_rank; ++p) {
        if (!assigned[p]) continue;
        int group = 0;
        switch (roles[p]) {
            case DimRole::Batch:
                group = 0;
                break;
            case DimRole::Free:
                group = transposed ? 2 : 1;
                break;
            case DimRole::Contracted:
                group = transposed ? 1 : 2;
                break;
        }
        if (group < max_group) return false;
        max_group = group;
    }
    return true;
}

inline bool check_group_order_B(const DimRole roles[MAX_RANK],
                                const bool assigned[MAX_RANK],
                                dim_t B_rank,
                                bool transposed) {
    // B normal group order: [Batch, Contracted, Free]
    // B transposed group order: [Batch, Free, Contracted]
    int max_group = -1;
    for (int32_t p = 0; p < B_rank; ++p) {
        if (!assigned[p]) continue;
        int group = 0;
        switch (roles[p]) {
            case DimRole::Batch:
                group = 0;
                break;
            case DimRole::Contracted:
                group = transposed ? 2 : 1;
                break;
            case DimRole::Free:
                group = transposed ? 1 : 2;
                break;
        }
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

inline GemmOperandLayout classify_gemm_layout_A(const ContractionConfig& config, dim_t A_rank) {
    DimRole roles[MAX_RANK];
    bool assigned[MAX_RANK] = {};

    for (int32_t i = 0; i < config.num_batch; ++i) {
        int32_t pos = config.batch_pos_A[i];
        roles[pos] = DimRole::Batch;
        assigned[pos] = true;
    }
    for (int32_t i = 0; i < config.num_free_A; ++i) {
        int32_t pos = config.free_pos_A[i];
        roles[pos] = DimRole::Free;
        assigned[pos] = true;
    }
    for (int32_t i = 0; i < config.num_contracted; ++i) {
        int32_t pos = config.contracted_pos_A[i];
        roles[pos] = DimRole::Contracted;
        assigned[pos] = true;
    }

    // Batch dims must be in config-index order for linearized batch iteration
    if (!positions_in_order(config.batch_pos_A, config.num_batch))
        return GemmOperandLayout::Interleaved;

    // Contracted dims must be in config-index order so flattened K matches
    // between operands (permutation functions produce config-index order)
    if (!positions_in_order(config.contracted_pos_A, config.num_contracted))
        return GemmOperandLayout::Interleaved;

    bool normal_ok = check_group_order_A(roles, assigned, A_rank, false);
    bool transposed_ok = check_group_order_A(roles, assigned, A_rank, true);

    if (normal_ok) return GemmOperandLayout::Normal;
    if (transposed_ok) return GemmOperandLayout::Transposed;
    return GemmOperandLayout::Interleaved;
}

inline GemmOperandLayout classify_gemm_layout_B(const ContractionConfig& config, dim_t B_rank) {
    DimRole roles[MAX_RANK];
    bool assigned[MAX_RANK] = {};

    for (int32_t i = 0; i < config.num_batch; ++i) {
        int32_t pos = config.batch_pos_B[i];
        roles[pos] = DimRole::Batch;
        assigned[pos] = true;
    }
    for (int32_t i = 0; i < config.num_contracted; ++i) {
        int32_t pos = config.contracted_pos_B[i];
        roles[pos] = DimRole::Contracted;
        assigned[pos] = true;
    }
    for (int32_t i = 0; i < config.num_free_B; ++i) {
        int32_t pos = config.free_pos_B[i];
        roles[pos] = DimRole::Free;
        assigned[pos] = true;
    }

    // Batch dims must be in config-index order for linearized batch iteration
    if (!positions_in_order(config.batch_pos_B, config.num_batch))
        return GemmOperandLayout::Interleaved;

    // Contracted dims must be in config-index order so flattened K matches
    if (!positions_in_order(config.contracted_pos_B, config.num_contracted))
        return GemmOperandLayout::Interleaved;

    bool normal_ok = check_group_order_B(roles, assigned, B_rank, false);
    bool transposed_ok = check_group_order_B(roles, assigned, B_rank, true);

    if (normal_ok) return GemmOperandLayout::Normal;
    if (transposed_ok) return GemmOperandLayout::Transposed;
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

// Check that A's mode positions follow the required [B..., F..., C...] order.
// Returns true if compatible, false if interleaved.
inline bool gemm_layout_compatible_A(const ContractionConfig& config, dim_t A_rank) {
    return classify_gemm_layout_A(config, A_rank) == GemmOperandLayout::Normal;
}

// Check that B's mode positions follow the required [B..., C..., F...] order.
// Returns true if compatible, false if interleaved.
inline bool gemm_layout_compatible_B(const ContractionConfig& config, dim_t B_rank) {
    return classify_gemm_layout_B(config, B_rank) == GemmOperandLayout::Normal;
}

// Validate that both operands have GEMM-compatible layouts.
// Throws std::runtime_error if either operand has interleaved dimension groups.
inline void validate_gemm_layout(const ContractionConfig& config,
                                 dim_t A_rank, dim_t B_rank) {
    GemmOperandLayout a_layout = classify_gemm_layout_A(config, A_rank);
    GemmOperandLayout b_layout = classify_gemm_layout_B(config, B_rank);
    bool a_ok = (a_layout != GemmOperandLayout::Interleaved);
    bool b_ok = (b_layout != GemmOperandLayout::Interleaved);

    if (!a_ok && !b_ok) {
        throw std::runtime_error(
            "CUTLASS GEMM layout incompatible: both A and B have interleaved "
            "dimension groups. Supported pass-through groupings are "
            "A: [batch..., free..., contracted...] or [batch..., contracted..., free...], "
            "B: [batch..., contracted..., free...] or [batch..., free..., contracted...].");
    }
    if (!a_ok) {
        throw std::runtime_error(
            "CUTLASS GEMM layout incompatible: A has interleaved dimension groups. "
            "Supported A groupings are [batch..., free..., contracted...] "
            "or [batch..., contracted..., free...].");
    }
    if (!b_ok) {
        throw std::runtime_error(
            "CUTLASS GEMM layout incompatible: B has interleaved dimension groups. "
            "Supported B groupings are [batch..., contracted..., free...] "
            "or [batch..., free..., contracted...].");
    }
}

// Compute permutation to reorder A's dims to one of the GEMM-compatible groupings.
// perm[src_dim] = dst_dim in the selected ordering.
inline std::vector<int32_t> compute_gemm_permutation_A_for_layout(
    const ContractionConfig& config,
    dim_t A_rank,
    GemmOperandLayout target_layout) {
    if (target_layout == GemmOperandLayout::Interleaved) {
        throw std::runtime_error("A target layout cannot be interleaved");
    }

    std::vector<int32_t> perm(A_rank);
    int32_t dst = 0;
    for (int32_t i = 0; i < config.num_batch; ++i)
        perm[config.batch_pos_A[i]] = dst++;
    if (target_layout == GemmOperandLayout::Normal) {
        for (int32_t i = 0; i < config.num_free_A; ++i)
            perm[config.free_pos_A[i]] = dst++;
        for (int32_t i = 0; i < config.num_contracted; ++i)
            perm[config.contracted_pos_A[i]] = dst++;
    } else {
        for (int32_t i = 0; i < config.num_contracted; ++i)
            perm[config.contracted_pos_A[i]] = dst++;
        for (int32_t i = 0; i < config.num_free_A; ++i)
            perm[config.free_pos_A[i]] = dst++;
    }
    return perm;
}

// Compute permutation to reorder B's dims to one of the GEMM-compatible groupings.
// perm[src_dim] = dst_dim in the selected ordering.
inline std::vector<int32_t> compute_gemm_permutation_B_for_layout(
    const ContractionConfig& config,
    dim_t B_rank,
    GemmOperandLayout target_layout) {
    if (target_layout == GemmOperandLayout::Interleaved) {
        throw std::runtime_error("B target layout cannot be interleaved");
    }

    std::vector<int32_t> perm(B_rank);
    int32_t dst = 0;
    for (int32_t i = 0; i < config.num_batch; ++i)
        perm[config.batch_pos_B[i]] = dst++;
    if (target_layout == GemmOperandLayout::Normal) {
        for (int32_t i = 0; i < config.num_contracted; ++i)
            perm[config.contracted_pos_B[i]] = dst++;
        for (int32_t i = 0; i < config.num_free_B; ++i)
            perm[config.free_pos_B[i]] = dst++;
    } else {
        for (int32_t i = 0; i < config.num_free_B; ++i)
            perm[config.free_pos_B[i]] = dst++;
        for (int32_t i = 0; i < config.num_contracted; ++i)
            perm[config.contracted_pos_B[i]] = dst++;
    }
    return perm;
}

// Backward-compatible wrappers for normal (NN) grouped order.
inline std::vector<int32_t> compute_gemm_permutation_A(
    const ContractionConfig& config, dim_t A_rank) {
    return compute_gemm_permutation_A_for_layout(config, A_rank, GemmOperandLayout::Normal);
}

inline std::vector<int32_t> compute_gemm_permutation_B(
    const ContractionConfig& config, dim_t B_rank) {
    return compute_gemm_permutation_B_for_layout(config, B_rank, GemmOperandLayout::Normal);
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

inline PermutationCandidate make_candidate_A(const ContractionConfig& config,
                                             dim_t A_rank,
                                             GemmOperandLayout target_layout) {
    PermutationCandidate candidate;
    candidate.target_layout = target_layout;
    candidate.perm = compute_gemm_permutation_A_for_layout(config, A_rank, target_layout);
    candidate.moved_dims = count_moved_dims(candidate.perm);
    candidate.unmoved_tail_dims = count_unmoved_tail_dims(candidate.perm);
    return candidate;
}

inline PermutationCandidate make_candidate_B(const ContractionConfig& config,
                                             dim_t B_rank,
                                             GemmOperandLayout target_layout) {
    PermutationCandidate candidate;
    candidate.target_layout = target_layout;
    candidate.perm = compute_gemm_permutation_B_for_layout(config, B_rank, target_layout);
    candidate.moved_dims = count_moved_dims(candidate.perm);
    candidate.unmoved_tail_dims = count_unmoved_tail_dims(candidate.perm);
    return candidate;
}

inline GemmLayoutPlan::OperandPlan plan_operand_A(const ContractionConfig& config,
                                                  dim_t A_rank,
                                                  GemmOperandLayout current_layout) {
    GemmLayoutPlan::OperandPlan out;
    out.original_layout = current_layout;
    out.target_layout = current_layout;
    out.transpose = (current_layout == GemmOperandLayout::Interleaved);
    if (!out.transpose) {
        return out;
    }

    PermutationCandidate normal = make_candidate_A(config, A_rank, GemmOperandLayout::Normal);
    PermutationCandidate transposed = make_candidate_A(
        config, A_rank, GemmOperandLayout::Transposed);
    const PermutationCandidate& best =
        is_better_candidate(transposed, normal) ? transposed : normal;

    out.target_layout = best.target_layout;
    out.perm = best.perm;
    out.unmoved_tail_dims = best.unmoved_tail_dims;
    out.dispatch_kind = TransposeDispatchKind::GenericPermute;
    return out;
}

inline GemmLayoutPlan::OperandPlan plan_operand_B(const ContractionConfig& config,
                                                  dim_t B_rank,
                                                  GemmOperandLayout current_layout) {
    GemmLayoutPlan::OperandPlan out;
    out.original_layout = current_layout;
    out.target_layout = current_layout;
    out.transpose = (current_layout == GemmOperandLayout::Interleaved);
    if (!out.transpose) {
        return out;
    }

    PermutationCandidate normal = make_candidate_B(config, B_rank, GemmOperandLayout::Normal);
    PermutationCandidate transposed = make_candidate_B(
        config, B_rank, GemmOperandLayout::Transposed);
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
    GemmLayoutPlan plan;
    plan.layout_A = classify_gemm_layout_A(config, A_rank);
    plan.layout_B = classify_gemm_layout_B(config, B_rank);

    plan.plan_A = plan_operand_A(config, A_rank, plan.layout_A);
    plan.plan_B = plan_operand_B(config, B_rank, plan.layout_B);
    plan.transpose_A = plan.plan_A.transpose;
    plan.transpose_B = plan.plan_B.transpose;
    plan.mode_after_transpose =
        gemm_mode_from_layouts(plan.plan_A.target_layout, plan.plan_B.target_layout);
    return plan;
}

} // namespace bsptc
