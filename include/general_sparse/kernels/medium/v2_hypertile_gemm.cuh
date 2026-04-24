#pragma once

#include "../../pipeline_descs.hpp"
#include "../transpose/nd_hypertile.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

// Fused hypertile medium GEMM: scatter-gather A/B loads via GridDivmod,
// eliminating the separate transpose pass for interleaved layouts.
// One CTA per problem (BatchExpander splits batch_size>1 on host side).
//
// Key design:
// - Template on R (max operand rank) and EPT (elements per thread for loads)
// - Fixed-size register offset maps: read_off_A[EPT], read_off_B[EPT]
// - Two-level K tiling: K_tile (SMEM compute tile) and K_load (load sub-tile)
// - K_tile sized to fill SMEM; K_load sized to fit EPT register budget
// - K load odometer advances through K_load sub-iterations within each K tile
namespace bsptc {

// ============================================================================
// MediumHypertileDesc<R>: compact descriptor for fused scatter-gather GEMM.
// R = max(A_rank, B_rank) — all arrays sized to R entries.
// ============================================================================

static constexpr int MHTD_MAX_RANK = 6;   // max squashed dims per operand
static constexpr int MHTD_MAX_K_RANK = 4; // max K dims for odometer

template <int R>
struct MediumHypertileDesc {
    // --- Pointers and dimensions ---
    const void* A;
    const void* B;
    void* C;
    dim_t M, N;
    int8_t use_atomic;
    int8_t beta_mode;  // 0=zero-init C then accumulate, 1=accumulate only (C already initialized)

    // --- A loader: GridDivmod over (free_M, K_load) dims ---
    // Built for K_load sub-tile, not full K_tile.
    int8_t A_rank;
    GridDivmod A_divmod[R];
    uint32_t   A_stride[R];
    int32_t    A_load_volume;   // M * K_load_volume

    // --- B loader: GridDivmod over (free_N, K_load) dims ---
    int8_t B_rank;
    GridDivmod B_divmod[R];
    uint32_t   B_stride[R];
    int32_t    B_load_volume;   // N * K_load_volume

    // --- K outer odometer (across K tiles) ---
    int8_t  K_grid_rank;
    int32_t K_total_tiles;
    int32_t K_tile_volume;
    uint32_t K_grid_step_A[MHTD_MAX_K_RANK];
    uint32_t K_grid_step_B[MHTD_MAX_K_RANK];
    int32_t K_grid_shape[MHTD_MAX_K_RANK];

    // --- K load odometer (sub-iterations within one K tile) ---
    int8_t  K_load_grid_rank;
    int32_t K_load_num_iters;      // total load sub-iterations per K tile
    int32_t K_load_volume;         // product of k_load_shape dims
    uint32_t K_load_step_A[MHTD_MAX_K_RANK];   // pointer advance per load dim
    uint32_t K_load_step_B[MHTD_MAX_K_RANK];
    int32_t K_load_grid_shape[MHTD_MAX_K_RANK]; // k_tile_shape[d] / k_load_shape[d]
    int32_t K_load_smem_step[MHTD_MAX_K_RANK];  // SMEM k-offset advance per load dim

    // --- Fast divmod for hot loops ---
    GridDivmod K_load_divmod;  // divmod by K_load_volume for load index → (m/n, k_local)
    GridDivmod N_divmod;       // divmod by N for compute index decomposition

    // --- K-edge bounds (for partial last K tile) ---
    int32_t K_raw_dim[MHTD_MAX_K_RANK];
    int32_t K_raw_tile_shape[MHTD_MAX_K_RANK];
    int8_t  K_raw_num_dims;
    int8_t  K_grid_to_raw[MHTD_MAX_K_RANK];

    // --- Batch (used by BatchExpander for pointer patching) ---
    dim_t batch_size;
    uint32_t batch_stride_a, batch_stride_b, batch_stride_c;
};

// Max-rank descriptor for host-side building (before packing to R).
using MediumHypertileDescMax = MediumHypertileDesc<MHTD_MAX_RANK>;

// Pack a max-rank descriptor to a compact R-sized descriptor.
template <int R>
MediumHypertileDesc<R> pack_mht_desc(const MediumHypertileDescMax& src) {
    MediumHypertileDesc<R> dst{};
    dst.A = src.A; dst.B = src.B; dst.C = src.C;
    dst.M = src.M; dst.N = src.N;
    dst.use_atomic = src.use_atomic;
    dst.beta_mode = src.beta_mode;
    dst.A_rank = src.A_rank;
    dst.A_load_volume = src.A_load_volume;
    dst.B_rank = src.B_rank;
    dst.B_load_volume = src.B_load_volume;
    dst.K_grid_rank = src.K_grid_rank;
    dst.K_total_tiles = src.K_total_tiles;
    dst.K_tile_volume = src.K_tile_volume;
    for (int i = 0; i < R; ++i) {
        dst.A_divmod[i] = src.A_divmod[i];
        dst.A_stride[i] = src.A_stride[i];
        dst.B_divmod[i] = src.B_divmod[i];
        dst.B_stride[i] = src.B_stride[i];
    }
    for (int i = 0; i < MHTD_MAX_K_RANK; ++i) {
        dst.K_grid_step_A[i] = src.K_grid_step_A[i];
        dst.K_grid_step_B[i] = src.K_grid_step_B[i];
        dst.K_grid_shape[i] = src.K_grid_shape[i];
    }
    // K load odometer
    dst.K_load_grid_rank = src.K_load_grid_rank;
    dst.K_load_num_iters = src.K_load_num_iters;
    dst.K_load_volume = src.K_load_volume;
    for (int i = 0; i < MHTD_MAX_K_RANK; ++i) {
        dst.K_load_step_A[i] = src.K_load_step_A[i];
        dst.K_load_step_B[i] = src.K_load_step_B[i];
        dst.K_load_grid_shape[i] = src.K_load_grid_shape[i];
        dst.K_load_smem_step[i] = src.K_load_smem_step[i];
    }
    dst.K_load_divmod = src.K_load_divmod;
    dst.N_divmod = src.N_divmod;
    dst.K_raw_num_dims = src.K_raw_num_dims;
    for (int i = 0; i < MHTD_MAX_K_RANK; ++i) {
        dst.K_raw_dim[i] = src.K_raw_dim[i];
        dst.K_raw_tile_shape[i] = src.K_raw_tile_shape[i];
        dst.K_grid_to_raw[i] = src.K_grid_to_raw[i];
    }
    dst.batch_size = src.batch_size;
    dst.batch_stride_a = src.batch_stride_a;
    dst.batch_stride_b = src.batch_stride_b;
    dst.batch_stride_c = src.batch_stride_c;
    return dst;
}

static_assert(std::is_trivially_copyable_v<MediumHypertileDescMax>);

// ============================================================================
// Per-rank group for GPU upload and dispatch.
// ============================================================================

struct MHTRankSlice {
    int rank;
    int num_problems;
    int max_K_tile_volume;
    int max_smem_elems;  // max per-problem (M*(K+1) + N*(K+1))
    std::vector<uint8_t> desc_bytes;  // type-erased MediumHypertileDesc<R> array
};

// Group max-rank descriptors by max(A_rank, B_rank) into per-rank slices.
inline std::vector<MHTRankSlice> group_mht_by_rank(
    const std::vector<MediumHypertileDescMax>& descs)
{
    // Bucket by rank
    std::map<int, std::vector<int>> rank_indices;
    for (int i = 0; i < static_cast<int>(descs.size()); ++i) {
        int r = std::max(static_cast<int>(descs[i].A_rank),
                         static_cast<int>(descs[i].B_rank));
        if (r < 2) r = 2;  // minimum rank 2 for template
        rank_indices[r].push_back(i);
    }

    std::vector<MHTRankSlice> slices;

    auto pack_rank = [&](auto rank_tag, int rank, const std::vector<int>& indices) {
        constexpr int R = decltype(rank_tag)::value;
        MHTRankSlice slice;
        slice.rank = rank;
        slice.num_problems = static_cast<int>(indices.size());
        slice.max_K_tile_volume = 0;
        slice.max_smem_elems = 0;

        slice.desc_bytes.resize(indices.size() * sizeof(MediumHypertileDesc<R>));
        auto* dst = reinterpret_cast<MediumHypertileDesc<R>*>(slice.desc_bytes.data());
        for (size_t j = 0; j < indices.size(); ++j) {
            dst[j] = pack_mht_desc<R>(descs[indices[j]]);
            const auto& d = descs[indices[j]];
            if (d.K_tile_volume > slice.max_K_tile_volume)
                slice.max_K_tile_volume = d.K_tile_volume;
            int stride = d.K_tile_volume + 1;
            int smem_elems = d.M * stride + d.N * stride;
            if (smem_elems > slice.max_smem_elems)
                slice.max_smem_elems = smem_elems;
        }
        slices.push_back(std::move(slice));
    };

    for (auto& [rank, indices] : rank_indices) {
        switch (rank) {
            case 2: pack_rank(std::integral_constant<int,2>{}, rank, indices); break;
            case 3: pack_rank(std::integral_constant<int,3>{}, rank, indices); break;
            case 4: pack_rank(std::integral_constant<int,4>{}, rank, indices); break;
            case 5: pack_rank(std::integral_constant<int,5>{}, rank, indices); break;
            case 6: pack_rank(std::integral_constant<int,6>{}, rank, indices); break;
            default: pack_rank(std::integral_constant<int,MHTD_MAX_RANK>{}, rank, indices); break;
        }
    }
    return slices;
}

// ============================================================================
// Kernel: fused hypertile medium GEMM, templated on R and EPT
// ============================================================================

// Compute scattered global offset from flat linear index using GridDivmod.
// Template on R for compile-time unrolled divmod chain.
template <int R>
__device__ __forceinline__
uint32_t ht_gemm_offset(int idx, int rank,
                        const GridDivmod divmod[],
                        const uint32_t stride[]) {
    uint32_t urem = static_cast<uint32_t>(idx);
    uint32_t off = 0;
    #pragma unroll
    for (int i = 0; i < R - 1; ++i) {
        if (i < rank - 1) {
            uint32_t q, r;
            divmod[i].divmod(urem, q, r);
            urem = q;
            off += r * stride[i];
        }
    }
    // Last dim: no divmod needed
    off += urem * stride[rank - 1];
    return off;
}

// Max C elements per thread (M*N < 4096, BLOCK_SIZE=128 → max 32).
static constexpr int MHT_MAX_C_PER_THREAD = 32;

template <typename T, int R, int EPT = 4, int BLOCK_SIZE = 128>
__global__ void medium_hypertile_gemm_kernel(
    const MediumHypertileDesc<R>* __restrict__ descs)
{
    // Cooperative load of descriptor into shared memory
    __shared__ MediumHypertileDesc<R> s_prob;
    {
        constexpr int WORDS = (sizeof(MediumHypertileDesc<R>) + sizeof(uint32_t) - 1)
                              / sizeof(uint32_t);
        auto* dst_w = reinterpret_cast<uint32_t*>(&s_prob);
        const auto* src_w = reinterpret_cast<const uint32_t*>(&descs[blockIdx.x]);
        for (int i = threadIdx.x; i < WORDS; i += BLOCK_SIZE) {
            dst_w[i] = src_w[i];
        }
    }
    __syncthreads();

    const int M = s_prob.M, N = s_prob.N;
    const int K_tile_vol = s_prob.K_tile_volume;
    const int A_load_vol = s_prob.A_load_volume;
    const int B_load_vol = s_prob.B_load_volume;
    const int total_C = M * N;
    const int tid = threadIdx.x;

    const T* __restrict__ A = static_cast<const T*>(s_prob.A);
    const T* __restrict__ B = static_cast<const T*>(s_prob.B);
    T* __restrict__ C = static_cast<T*>(s_prob.C);

    // Dynamic SMEM layout: sA[M * sA_stride] + sB[N * sB_stride]
    extern __shared__ char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);
    const int sA_stride = K_tile_vol + 1;
    const int sB_stride = K_tile_vol + 1;
    T* sA = smem;
    T* sB = smem + M * sA_stride;

    // Warp-uniform step counts: full_steps has all threads active (no branch),
    // tail step has only some threads active (one branch outside the main loop).
    const int full_steps_a = A_load_vol / BLOCK_SIZE;  // uniform: no thread divergence
    const int tail_a = A_load_vol % BLOCK_SIZE;         // threads with tid < tail participate
    const int full_steps_b = B_load_vol / BLOCK_SIZE;
    const int tail_b = B_load_vol % BLOCK_SIZE;

    // Precompute A load maps: global read offset + SMEM write index.
    // These depend only on (tid, step) and are invariant across K steps.
    uint32_t read_off_A[EPT];
    uint16_t smem_idx_A[EPT];   // m * sA_stride + k_local (add load_smem_base at use site)
    #pragma unroll
    for (int step = 0; step < EPT; ++step) {
        if (step < full_steps_a || (step == full_steps_a && tid < tail_a)) {
            read_off_A[step] = ht_gemm_offset<R>(
                tid + step * BLOCK_SIZE, s_prob.A_rank, s_prob.A_divmod, s_prob.A_stride);
            uint32_t m_u, k_local;
            s_prob.K_load_divmod.divmod(static_cast<uint32_t>(tid + step * BLOCK_SIZE), m_u, k_local);
            smem_idx_A[step] = m_u * sA_stride + k_local;
        }
    }

    // Precompute B load maps: global read offset + SMEM write index.
    uint32_t read_off_B[EPT];
    uint16_t smem_idx_B[EPT];   // n * sB_stride + k_local
    #pragma unroll
    for (int step = 0; step < EPT; ++step) {
        if (step < full_steps_b || (step == full_steps_b && tid < tail_b)) {
            read_off_B[step] = ht_gemm_offset<R>(
                tid + step * BLOCK_SIZE, s_prob.B_rank, s_prob.B_divmod, s_prob.B_stride);
            uint32_t n_u, k_local;
            s_prob.K_load_divmod.divmod(static_cast<uint32_t>(tid + step * BLOCK_SIZE), n_u, k_local);
            smem_idx_B[step] = n_u * sB_stride + k_local;
        }
    }

    // Zero-initialize output region (replaces register accumulator)
    // Skip when beta_mode=1 (C already initialized, accumulate onto existing values)
    if (s_prob.beta_mode == 0) {
        for (int i = tid; i < total_C; i += BLOCK_SIZE)
            C[i] = static_cast<T>(0);
    }

    // K outer odometer: stepping through K tiles
    int64_t k_base_A = 0, k_base_B = 0;
    int32_t k_raw_coord[MHTD_MAX_K_RANK] = {0};

    for (int k_step = 0; k_step < s_prob.K_total_tiles; ++k_step) {
        // Precompute allowed_k[d] for edge detection
        int32_t allowed_k[MHTD_MAX_K_RANK];
        int k_eff = 1;
        bool is_k_edge = false;
        for (int d = 0; d < s_prob.K_raw_num_dims; ++d) {
            int rem = s_prob.K_raw_dim[d] - k_raw_coord[d] * s_prob.K_raw_tile_shape[d];
            allowed_k[d] = min(rem, static_cast<int>(s_prob.K_raw_tile_shape[d]));
            k_eff *= allowed_k[d];
            if (allowed_k[d] < s_prob.K_raw_tile_shape[d]) is_k_edge = true;
        }
        if (!is_k_edge || s_prob.K_raw_num_dims > 1) k_eff = K_tile_vol;

        if (is_k_edge) {
            // === Edge path: zero SMEM, load with K bounds checks ===
            for (int i = tid; i < M * sA_stride; i += BLOCK_SIZE) sA[i] = T(0);
            for (int i = tid; i < N * sB_stride; i += BLOCK_SIZE) sB[i] = T(0);
            __syncthreads();

            int64_t load_base_A = 0, load_base_B = 0;
            int32_t load_smem_base = 0;
            int32_t load_coord[MHTD_MAX_K_RANK] = {0};
            // Helper lambda for edge-checked store (uses precomputed smem_idx)
            auto edge_store_a = [&](int step, int32_t lsb) {
                int32_t smem_pos = static_cast<int32_t>(smem_idx_A[step]) + lsb;
                // Extract k_smem from smem_pos: it's (m*sA_stride + lsb + k_local),
                // k component is smem_pos % sA_stride
                int32_t k_smem = smem_pos % sA_stride;
                bool k_valid = true;
                uint32_t krem = static_cast<uint32_t>(k_smem);
                for (int d = 0; d < s_prob.K_raw_num_dims; ++d) {
                    uint32_t ts = static_cast<uint32_t>(s_prob.K_raw_tile_shape[d]);
                    uint32_t kc = krem % ts;
                    krem /= ts;
                    if (kc >= static_cast<uint32_t>(allowed_k[d])) { k_valid = false; break; }
                }
                if (k_valid)
                    sA[smem_pos] = A[k_base_A + load_base_A + read_off_A[step]];
            };
            auto edge_store_b = [&](int step, int32_t lsb) {
                int32_t smem_pos = static_cast<int32_t>(smem_idx_B[step]) + lsb;
                int32_t k_smem = smem_pos % sB_stride;
                bool k_valid = true;
                uint32_t krem = static_cast<uint32_t>(k_smem);
                for (int d = 0; d < s_prob.K_raw_num_dims; ++d) {
                    uint32_t ts = static_cast<uint32_t>(s_prob.K_raw_tile_shape[d]);
                    uint32_t kc = krem % ts;
                    krem /= ts;
                    if (kc >= static_cast<uint32_t>(allowed_k[d])) { k_valid = false; break; }
                }
                if (k_valid)
                    sB[smem_pos] = B[k_base_B + load_base_B + read_off_B[step]];
            };
            for (int li = 0; li < s_prob.K_load_num_iters; ++li) {
                for (int step = 0; step < full_steps_a; ++step)
                    edge_store_a(step, load_smem_base);
                if (tail_a && tid < tail_a)
                    edge_store_a(full_steps_a, load_smem_base);
                for (int step = 0; step < full_steps_b; ++step)
                    edge_store_b(step, load_smem_base);
                if (tail_b && tid < tail_b)
                    edge_store_b(full_steps_b, load_smem_base);
                for (int d = s_prob.K_load_grid_rank - 1; d >= 0; --d) {
                    load_coord[d]++;
                    if (load_coord[d] < s_prob.K_load_grid_shape[d]) {
                        load_base_A += s_prob.K_load_step_A[d];
                        load_base_B += s_prob.K_load_step_B[d];
                        load_smem_base += s_prob.K_load_smem_step[d];
                        break;
                    } else {
                        load_coord[d] = 0;
                        load_base_A -= s_prob.K_load_step_A[d] * (s_prob.K_load_grid_shape[d] - 1);
                        load_base_B -= s_prob.K_load_step_B[d] * (s_prob.K_load_grid_shape[d] - 1);
                        load_smem_base -= s_prob.K_load_smem_step[d] * (s_prob.K_load_grid_shape[d] - 1);
                    }
                }
            }
        } else {
            // === Fast path: no edge checks, minimal branches ===
            int64_t load_base_A = 0, load_base_B = 0;
            int32_t load_smem_base = 0;
            int32_t load_coord[MHTD_MAX_K_RANK] = {0};
            for (int li = 0; li < s_prob.K_load_num_iters; ++li) {
                // A: full steps (all threads active, no branch)
                for (int step = 0; step < full_steps_a; ++step) {
                    sA[smem_idx_A[step] + load_smem_base] =
                        A[k_base_A + load_base_A + read_off_A[step]];
                }
                // A: tail step (partial threads)
                if (tail_a && tid < tail_a) {
                    sA[smem_idx_A[full_steps_a] + load_smem_base] =
                        A[k_base_A + load_base_A + read_off_A[full_steps_a]];
                }
                // B: full steps
                for (int step = 0; step < full_steps_b; ++step) {
                    sB[smem_idx_B[step] + load_smem_base] =
                        B[k_base_B + load_base_B + read_off_B[step]];
                }
                // B: tail step
                if (tail_b && tid < tail_b) {
                    sB[smem_idx_B[full_steps_b] + load_smem_base] =
                        B[k_base_B + load_base_B + read_off_B[full_steps_b]];
                }
                for (int d = s_prob.K_load_grid_rank - 1; d >= 0; --d) {
                    load_coord[d]++;
                    if (load_coord[d] < s_prob.K_load_grid_shape[d]) {
                        load_base_A += s_prob.K_load_step_A[d];
                        load_base_B += s_prob.K_load_step_B[d];
                        load_smem_base += s_prob.K_load_smem_step[d];
                        break;
                    } else {
                        load_coord[d] = 0;
                        load_base_A -= s_prob.K_load_step_A[d] * (s_prob.K_load_grid_shape[d] - 1);
                        load_base_B -= s_prob.K_load_step_B[d] * (s_prob.K_load_grid_shape[d] - 1);
                        load_smem_base -= s_prob.K_load_smem_step[d] * (s_prob.K_load_grid_shape[d] - 1);
                    }
                }
            }
        }

        __syncthreads();

        // Accumulate directly into global C
        for (int c_idx = tid; c_idx < total_C; c_idx += BLOCK_SIZE) {
            uint32_t m_u, n_u;
            s_prob.N_divmod.divmod(static_cast<uint32_t>(c_idx), m_u, n_u);
            const T* sA_row = sA + m_u * sA_stride;
            const T* sB_row = sB + n_u * sB_stride;
            T sum = static_cast<T>(0);
            for (int kl = 0; kl < k_eff; ++kl) {
                sum += sA_row[kl] * sB_row[kl];
            }
            C[c_idx] += sum;
        }

        __syncthreads();

        // Advance K outer odometer
        for (int d = s_prob.K_grid_rank - 1; d >= 0; --d) {
            int raw_d = s_prob.K_grid_to_raw[d];
            k_raw_coord[raw_d]++;
            if (k_raw_coord[raw_d] < s_prob.K_grid_shape[d]) {
                k_base_A += s_prob.K_grid_step_A[d];
                k_base_B += s_prob.K_grid_step_B[d];
                break;
            } else {
                k_raw_coord[raw_d] = 0;
                k_base_A -= s_prob.K_grid_step_A[d] * (s_prob.K_grid_shape[d] - 1);
                k_base_B -= s_prob.K_grid_step_B[d] * (s_prob.K_grid_shape[d] - 1);
            }
        }
    }
}

// Host-side launch helper for a single rank slice.
template <typename T, int R, int EPT = 4, int BLOCK_SIZE = 128>
void launch_medium_hypertile_gemm_rank(
    const void* d_descs,
    int num_problems,
    int smem_bytes,
    cudaStream_t stream)
{
    if (num_problems == 0) return;

    constexpr int DEFAULT_SMEM_LIMIT = 48 * 1024;
    if (smem_bytes > DEFAULT_SMEM_LIMIT) {
        auto kern = medium_hypertile_gemm_kernel<T, R, EPT, BLOCK_SIZE>;
        cudaFuncSetAttribute(kern,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    medium_hypertile_gemm_kernel<T, R, EPT, BLOCK_SIZE>
        <<<num_problems, BLOCK_SIZE, smem_bytes, stream>>>(
            reinterpret_cast<const MediumHypertileDesc<R>*>(d_descs));
}

// Dispatch helper: launch for all rank slices.
template <typename T, int BLOCK_SIZE = 128>
void launch_medium_hypertile_gemm(
    const std::vector<MHTRankSlice>& slices,
    const std::vector<void*>& d_desc_ptrs,
    cudaStream_t stream)
{
    for (size_t s = 0; s < slices.size(); ++s) {
        const auto& sl = slices[s];
        if (sl.num_problems == 0) continue;

        int smem_bytes = sl.max_smem_elems * static_cast<int>(sizeof(T));

#define MHT_LAUNCH_R(RV) \
    launch_medium_hypertile_gemm_rank<T, RV, 4, BLOCK_SIZE>( \
        d_desc_ptrs[s], sl.num_problems, smem_bytes, stream)

        switch (sl.rank) {
            case 2: MHT_LAUNCH_R(2); break;
            case 3: MHT_LAUNCH_R(3); break;
            case 4: MHT_LAUNCH_R(4); break;
            case 5: MHT_LAUNCH_R(5); break;
            case 6: MHT_LAUNCH_R(6); break;
            default: MHT_LAUNCH_R(6); break;
        }
#undef MHT_LAUNCH_R
    }
}

// Legacy launch interface (used by pipeline.cuh).
template <typename T, int BLOCK_SIZE = 128>
void launch_medium_hypertile_gemm(
    const MediumHypertileDescMax* d_descs,
    int num_problems,
    int max_smem_elems,
    int /*max_K_tile_volume — unused, stride is per-problem*/,
    cudaStream_t stream = 0)
{
    int smem_bytes = max_smem_elems * static_cast<int>(sizeof(T));
    launch_medium_hypertile_gemm_rank<T, MHTD_MAX_RANK, 4, BLOCK_SIZE>(
        d_descs, num_problems, smem_bytes, stream);
}

} // namespace bsptc
