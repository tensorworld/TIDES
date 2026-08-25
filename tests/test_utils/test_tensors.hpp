#pragma once

// Shared test tensor definitions for pipeline verification tests.
// Each pair defines inline .bs-format tensor content for a specific test scenario.

namespace test_tensors {

// Basic 2-block diagonal matrix (ij / jk)
inline const char* MAT_A = R"(
# Tensor A (ij)
{2, 2}
2
{3, 3, 4, 4}
0, 0
1, 1
)";

inline const char* MAT_B = R"(
# Tensor B (jk)
{2, 2}
2
{4, 4, 3, 3}
0, 0
1, 1
)";

// Accumulation: multiple pairs map to same output block
inline const char* ACCUM_A = R"(
# Tensor A (ij)
{1, 2}
2
{2, 2, 2}
0, 0
0, 1
)";

inline const char* ACCUM_B = R"(
# Tensor B (jk)
{2, 1}
2
{2, 2, 2}
0, 0
1, 0
)";

// 3D tensors
inline const char* T3D_A = R"(
# Tensor A (ijk)
{2, 2, 2}
2
{3, 3, 2, 2, 2, 2}
0, 0, 0
1, 1, 1
)";

inline const char* T3D_B = R"(
# Tensor B (jkl)
{2, 2, 2}
2
{2, 2, 2, 2, 3, 3}
0, 0, 0
1, 1, 1
)";

// 4D GEMM-like
inline const char* T4D_A = R"(
# Tensor A (abcd)
{2, 2, 2, 2}
4
{2, 2, 2, 2, 2, 2, 2, 2}
0, 0, 0, 0
0, 1, 1, 0
1, 0, 0, 1
1, 1, 1, 1
)";

inline const char* T4D_B = R"(
# Tensor B (efcd)
{2, 2, 2, 2}
4
{2, 2, 2, 2, 2, 2, 2, 2}
0, 0, 0, 0
0, 1, 1, 0
1, 0, 0, 1
1, 1, 1, 1
)";

// Outer product (no contracted indices)
inline const char* OUTER_A = R"(
# Tensor A (ij)
{2, 2}
2
{2, 2, 3, 3}
0, 0
1, 1
)";

inline const char* OUTER_B = R"(
# Tensor B (kl)
{2, 2}
2
{2, 2, 3, 3}
0, 0
1, 1
)";

// Batch dimension
inline const char* BATCH_A = R"(
# Tensor A (bij)
{2, 2, 2}
3
{2, 2, 3, 3, 4, 4}
0, 0, 0
0, 1, 1
1, 0, 0
)";

inline const char* BATCH_B = R"(
# Tensor B (bjk)
{2, 2, 2}
3
{2, 2, 4, 4, 3, 3}
0, 0, 0
0, 1, 1
1, 0, 0
)";

// Multi-block sparse (6 blocks each, 4 sections per dim)
inline const char* MULTI_A = R"(
# Tensor A (ij)
{4, 4}
6
{3, 3, 3, 3, 4, 4, 4, 4}
0, 0
0, 2
1, 1
2, 0
2, 2
3, 3
)";

inline const char* MULTI_B = R"(
# Tensor B (jk)
{4, 4}
6
{4, 4, 4, 4, 3, 3, 3, 3}
0, 0
0, 2
1, 1
2, 0
2, 2
3, 3
)";

// Interleaved layout test cases: these violate CUTLASS GEMM dimension grouping.
// Small blocks (2-4 elements per section), 2 diagonal blocks each.

// Interleaved A: einsum jik,jkl->il
// A modes "jik": j=contracted(pos0), i=free(pos1), k=contracted(pos2) → interleaved
// B modes "jkl": j=contracted(pos0), k=contracted(pos1), l=free(pos2) → OK
inline const char* INTERLEAVED_A_A = R"(
# Tensor A (jik) - contracted/free/contracted interleaved
{2, 2, 2}
2
{3, 3, 2, 2, 4, 4}
0, 0, 0
1, 1, 1
)";

inline const char* INTERLEAVED_A_B = R"(
# Tensor B (jkl)
{2, 2, 2}
2
{3, 3, 4, 4, 2, 2}
0, 0, 0
1, 1, 1
)";

// Interleaved B: einsum ijk,ljk->il
// A modes "ijk": i=free(pos0), j=contracted(pos1), k=contracted(pos2) → OK
// B modes "ljk": l=free(pos0), j=contracted(pos1), k=contracted(pos2) → interleaved
//   B requires [contracted..., free...] but has [free, contracted, contracted]
inline const char* INTERLEAVED_B_A = R"(
# Tensor A (ijk)
{2, 2, 2}
2
{2, 2, 3, 3, 4, 4}
0, 0, 0
1, 1, 1
)";

inline const char* INTERLEAVED_B_B = R"(
# Tensor B (ljk) - free/contracted/contracted interleaved
{2, 2, 2}
2
{2, 2, 3, 3, 4, 4}
0, 0, 0
1, 1, 1
)";

// Batch not leading: einsum ibj,bjk->bik
// A modes "ibj": i=free(pos0), b=batch(pos1), j=contracted(pos2) → interleaved
// B modes "bjk": b=batch(pos0), j=contracted(pos1), k=free(pos2) → OK
inline const char* BATCH_NONLEAD_A = R"(
# Tensor A (ibj) - free/batch/contracted interleaved
{2, 2, 2}
2
{3, 3, 2, 2, 4, 4}
0, 0, 0
1, 1, 1
)";

inline const char* BATCH_NONLEAD_B = R"(
# Tensor B (bjk)
{2, 2, 2}
2
{2, 2, 4, 4, 3, 3}
0, 0, 0
1, 1, 1
)";

// Batch not leading in B: einsum bij,jbk->bik
// A modes "bij": b=batch(pos0), i=free(pos1), j=contracted(pos2) → OK
// B modes "jbk": j=contracted(pos0), b=batch(pos1), k=free(pos2) → interleaved
inline const char* BATCH_NONLEAD_B_SIDE_A = R"(
# Tensor A (bij) - batch leading
{2, 2, 2}
2
{2, 2, 3, 3, 4, 4}
0, 0, 0
1, 1, 1
)";

inline const char* BATCH_NONLEAD_B_SIDE_B = R"(
# Tensor B (jbk) - contracted/batch/free interleaved
{2, 2, 2}
2
{4, 4, 2, 2, 3, 3}
0, 0, 0
1, 1, 1
)";

// Reversed two-batch prefix: einsum abij,bajk->abik
// Batch labels lead in BOTH operands, but in different orders (ab vs ba);
// the batch-leftmost invariant requires identical canonical order, so the
// pipeline must reject. This is the shape the fuzzer hit at seed 120.
inline const char* BATCH_PREFIX_REVERSED_A = R"(
# Tensor A (abij) - batch prefix a,b
{2, 2, 2, 2}
2
{2, 2, 2, 2, 3, 3, 4, 4}
0, 0, 0, 0
1, 1, 1, 1
)";

inline const char* BATCH_PREFIX_REVERSED_B = R"(
# Tensor B (bajk) - batch prefix b,a (reversed)
{2, 2, 2, 2}
2
{2, 2, 2, 2, 4, 4, 3, 3}
0, 0, 0, 0
1, 1, 1, 1
)";

// --- Squashed-rank overflow tensors -----------------------------------------
// K-reversal shapes (muvwxy,yxwvun->mn): the 5 contracted dims u..y are
// contiguous in BOTH operands but in reversed order in B, so the dual squash
// (which needs both stride sets contiguous) keeps all 5 groups — exceeding
// WARP_MAX_DIMS=4 — and B's combined K+free volume rank is 6 > VOL_MAX_RANK=5.
// The partitioner's rank-eligibility check must reroute these to CUTLASS
// (whose B transpose squashes to 6 groups, well within HT_MAX_RANK). Before
// that check existed, the warp/volume desc builders overflowed fixed-capacity
// stack arrays.
inline const char* RANK_OVERFLOW_KREV_A_WARP = R"(
# Tensor A (muvwxy) - free m=2, contracted u..y contiguous, rank 6
{1, 1, 1, 1, 1, 1}
1
{2, 2, 2, 2, 2, 2}
0, 0, 0, 0, 0, 0
)";

inline const char* RANK_OVERFLOW_KREV_B_WARP = R"(
# Tensor B (yxwvun) - contracted reversed vs A, free n=2, rank 6
{1, 1, 1, 1, 1, 1}
1
{2, 2, 2, 2, 2, 2}
0, 0, 0, 0, 0, 0
)";

// Volume variant: m=n=4 → min(M,N)=4 skips the warp tier; volume by magnitude
// (M*K+N*K = 256), vetoed by rank (B combined 5+1 = 6 > VOL_MAX_RANK).
inline const char* RANK_OVERFLOW_KREV_A_VOL = R"(
# Tensor A (muvwxy) - free m=4, contracted u..y contiguous, rank 6
{1, 1, 1, 1, 1, 1}
1
{4, 2, 2, 2, 2, 2}
0, 0, 0, 0, 0, 0
)";

inline const char* RANK_OVERFLOW_KREV_B_VOL = R"(
# Tensor B (yxwvun) - contracted reversed vs A, free n=4, rank 6
{1, 1, 1, 1, 1, 1}
1
{2, 2, 2, 2, 2, 4}
0, 0, 0, 0, 0, 0
)";

// Fully-alternating shapes (aubvcwdxey): 5 free dims interleaved with 5
// contracted dims — squashed free AND K rank are both 5. Warp/volume can't
// hold them, AND the CUTLASS transpose would need 10 non-mergeable groups
// (> HT_MAX_RANK=8), so the pipeline cannot execute this contraction at all.
// Setup must fail loudly (it used to corrupt stack arrays instead).
inline const char* RANK_OVERFLOW_A = R"(
# Tensor A (aubvcwdxey) - free/contracted alternating, rank 10, all dims 2
{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
1
{2, 2, 2, 2, 2, 2, 2, 2, 2, 2}
0, 0, 0, 0, 0, 0, 0, 0, 0, 0
)";

inline const char* RANK_OVERFLOW_B_WARP = R"(
# Tensor B (uvwxyz) - 5 contracted + 1 free, rank 6, all dims 2
{1, 1, 1, 1, 1, 1}
1
{2, 2, 2, 2, 2, 2}
0, 0, 0, 0, 0, 0
)";

inline const char* RANK_OVERFLOW_B_VOL = R"(
# Tensor B (uvwxyzt) - 5 contracted + 2 free, rank 7, all dims 2
{1, 1, 1, 1, 1, 1, 1}
1
{2, 2, 2, 2, 2, 2, 2}
0, 0, 0, 0, 0, 0, 0
)";

// Output rank overflow: acegik,bdfhjk->abcdefghij has a rank-10 output —
// beyond MAX_RANK=8, which bounds ContractionConfig::internal_to_output and
// every downstream output-coord array. build_contraction_config must flag it
// and pipeline_plan reject it loudly (previously the i2o array overflowed
// into output_rank and the stack — silent memory corruption).
inline const char* OUT_PERM_OVERFLOW_A = R"(
# Tensor A (acegik) - 5 free + contracted k, rank 6, all dims 2
{1, 1, 1, 1, 1, 1}
1
{2, 2, 2, 2, 2, 2}
0, 0, 0, 0, 0, 0
)";

inline const char* OUT_PERM_OVERFLOW_B = R"(
# Tensor B (bdfhjk) - 5 free + contracted k, rank 6, all dims 2
{1, 1, 1, 1, 1, 1}
1
{2, 2, 2, 2, 2, 2}
0, 0, 0, 0, 0, 0
)";

// All-trivial output permute: ij,jk->ki where every i-section has size 1.
// Each output block is [i=1, k], so swapping to [k, i=1] moves no data —
// the permute stage must short-circuit (buffer handed out as-is) while the
// reported block coords are still permuted to output order.
inline const char* TRIV_PERM_A = R"(
# Tensor A (ij) - i sections all size 1
{3, 2}
6
{1, 1, 1, 8, 12}
0, 0
0, 1
1, 0
1, 1
2, 0
2, 1
)";

inline const char* TRIV_PERM_B = R"(
# Tensor B (jk) - j matches A's j sections
{2, 2}
4
{8, 12, 6, 10}
0, 0
0, 1
1, 0
1, 1
)";

// Large-block interleaved layout tests: blocks large enough to go through CUTLASS (M*N >= 16384).
// einsum: jik,jkl->il (A interleaved, B ok)
// A modes "jik": j(pos0)=contracted, i(pos1)=free, k(pos2)=contracted → [C, F, C] interleaved
// B modes "jkl": j(pos0)=contracted, k(pos1)=contracted, l(pos2)=free → [C, C, F] OK
// Block sizes: j=64, i=128, k=64 for A → M=128, K=64*64=4096, M*N=128*128=16384
//              j=64, k=64, l=128 for B → N=128, K=64*64=4096
inline const char* LARGE_INTERLEAVED_A_A = R"(
# Tensor A (jik) - large blocks, contracted/free/contracted interleaved
{1, 1, 1}
1
{64, 128, 64}
0, 0, 0
)";

inline const char* LARGE_INTERLEAVED_A_B = R"(
# Tensor B (jkl) - large blocks
{1, 1, 1}
1
{64, 64, 128}
0, 0, 0
)";

// Large-block interleaved B: einsum ijk,ljk->il
// A modes "ijk": i(pos0)=free, j(pos1)=contracted, k(pos2)=contracted → [F, C, C] OK
// B modes "ljk": l(pos0)=free, j(pos1)=contracted, k(pos2)=contracted → [F, C, C]
//   B requires [C..., F...] → interleaved
inline const char* LARGE_INTERLEAVED_B_A = R"(
# Tensor A (ijk) - large blocks
{1, 1, 1}
1
{128, 64, 64}
0, 0, 0
)";

inline const char* LARGE_INTERLEAVED_B_B = R"(
# Tensor B (ljk) - large blocks, free/contracted/contracted interleaved
{1, 1, 1}
1
{128, 64, 64}
0, 0, 0
)";

// Large-block batch not leading: einsum ibj,bjk->bik
// A modes "ibj": i(pos0)=free, b(pos1)=batch, j(pos2)=contracted → [F, B, C] interleaved
// B modes "bjk": b(pos0)=batch, j(pos1)=contracted, k(pos2)=free → [B, C, F] OK
inline const char* LARGE_BATCH_NONLEAD_A = R"(
# Tensor A (ibj) - large blocks, free/batch/contracted interleaved
{1, 1, 1}
1
{128, 2, 128}
0, 0, 0
)";

inline const char* LARGE_BATCH_NONLEAD_B = R"(
# Tensor B (bjk) - large blocks
{1, 1, 1}
1
{2, 128, 128}
0, 0, 0
)";

// Mixed dispatch: some pairs large enough for CUTLASS (M*N >= 16384), some small (naive).
// einsum ij,jk->ik with varying block sizes in free dimensions.
// A has 2 i-sections (200, 10) and 1 j-section (32): blocks (0,0)=200x32, (1,0)=10x32
// B has 1 j-section (32) and 2 k-sections (200, 10): blocks (0,0)=32x200, (0,1)=32x10
// Pairs: (A[0,0],B[0,0]) M=200,N=200 -> CUTLASS; three others M*N<16384 -> naive
inline const char* MIXED_DISPATCH_A = R"(
# Tensor A (ij) - mixed block sizes for CUTLASS/naive split
{2, 1}
2
{200, 10, 32}
0, 0
1, 0
)";

inline const char* MIXED_DISPATCH_B = R"(
# Tensor B (jk) - mixed block sizes for CUTLASS/naive split
{1, 2}
2
{32, 200, 10}
0, 0
0, 1
)";

// ===============================================================
// GEMM Layout Variant Tensors (TN/NT/TT)
// NN is the default: A=[batch, free, contracted], B=[batch, contracted, free]
// TN: A has [contracted..., free...] (transposed A)
// NT: B has [free..., contracted...] (transposed B)
// TT: both transposed
// ===============================================================

// --- TN matrix: ji,jk->ik  (A=[contracted, free]) ---
inline const char* TN_MAT_A = R"(
# Tensor A (ji)
{2, 2}
2
{4, 4, 3, 3}
0, 0
1, 1
)";

// --- NT matrix: ij,kj->ik  (B=[free, contracted]) ---
inline const char* NT_MAT_B = R"(
# Tensor B (kj)
{2, 2}
2
{3, 3, 4, 4}
0, 0
1, 1
)";

// --- TN batched: bji,bjk->bik  (A=[batch, contracted, free]) ---
inline const char* BATCH_TN_A = R"(
# Tensor A (bji)
{2, 2, 2}
3
{2, 2, 4, 4, 3, 3}
0, 0, 0
0, 1, 1
1, 0, 0
)";

// --- NT batched: bij,bkj->bik  (B=[batch, free, contracted]) ---
inline const char* BATCH_NT_B = R"(
# Tensor B (bkj)
{2, 2, 2}
3
{2, 2, 3, 3, 4, 4}
0, 0, 0
0, 1, 1
1, 0, 0
)";

// --- TN 3D: jki,jkl->il  (A=[contracted, contracted, free]) ---
// Contracted dims j=[3,3], k=[4,4] match INTERLEAVED_B pair and NT_3D_B.
inline const char* TN_3D_A = R"(
# Tensor A (jki)
{2, 2, 2}
2
{3, 3, 4, 4, 2, 2}
0, 0, 0
1, 1, 1
)";

inline const char* TN_3D_B = R"(
# Tensor B (jkl)
{2, 2, 2}
2
{3, 3, 4, 4, 2, 2}
0, 0, 0
1, 1, 1
)";

// NT 3D: ijk,ljk->il  — reuse INTERLEAVED_B_A (ijk) + INTERLEAVED_B_B (ljk)
// B has [free, contracted, contracted]: properly grouped but reversed → NT, not interleaved.

// TT 3D: jki,ljk->il  — use TN_3D_A + INTERLEAVED_B_B
// Contracted dims j=[3,3], k=[4,4] match between TN_3D_A and INTERLEAVED_B_B.

// --- Large-block TN matrix: ji,jk->ik  (M*N = 128*128 = 16384) ---
inline const char* LARGE_TN_MAT_A = R"(
# Tensor A (ji) - large blocks
{1, 1}
1
{32, 128}
0, 0
)";

inline const char* LARGE_TN_MAT_B = R"(
# Tensor B (jk) - large blocks
{1, 1}
1
{32, 128}
0, 0
)";

// --- Large-block NT matrix: ij,kj->ik ---
inline const char* LARGE_NT_MAT_A = R"(
# Tensor A (ij) - large blocks
{1, 1}
1
{128, 32}
0, 0
)";

inline const char* LARGE_NT_MAT_B = R"(
# Tensor B (kj) - large blocks
{1, 1}
1
{128, 32}
0, 0
)";

// Large TT matrix: ji,kj->ik — use LARGE_TN_MAT_A + LARGE_NT_MAT_B

// --- Large-block TN 3D: jki,jkl->il  (M=128, N=128, K=64*64=4096) ---
inline const char* LARGE_TN_3D_A = R"(
# Tensor A (jki) - large blocks
{1, 1, 1}
1
{64, 64, 128}
0, 0, 0
)";

inline const char* LARGE_TN_3D_B = R"(
# Tensor B (jkl) - large blocks
{1, 1, 1}
1
{64, 64, 128}
0, 0, 0
)";

// Large NT 3D: ijk,ljk->il — reuse LARGE_INTERLEAVED_B_A + LARGE_INTERLEAVED_B_B
// Large TT 3D: jki,ljk->il — use LARGE_TN_3D_A + LARGE_INTERLEAVED_B_B
// Contracted dims j=64, k=64 match.

// ===============================================================
// Must-Transpose Tensors (truly interleaved dim groups: [C,F,C])
// These cannot be handled by CUTLASS transpose flags alone;
// they require a physical transpose to reorder dimensions.
// ===============================================================

// B truly interleaved: einsum ijk,jlk->il
// A(ijk) = [F,C,C] → OK for A
// B(jlk) = [C,F,C] → truly interleaved, needs physical transpose
// Also used for "both interleaved": jik,jlk->il with INTERLEAVED_A_A as A.
// Contracted dims j=[3,3], k=[4,4] match INTERLEAVED_A_A and INTERLEAVED_B_A.
inline const char* INTERLEAVED_B_TENSOR = R"(
# Tensor B (jlk) - contracted/free/contracted truly interleaved
{2, 2, 2}
2
{3, 3, 2, 2, 4, 4}
0, 0, 0
1, 1, 1
)";

// Large-block B truly interleaved: j=64, l=128, k=64
// M*N = 128*128 = 16384, K = 64*64 = 4096
inline const char* LARGE_INTERLEAVED_B_TENSOR = R"(
# Tensor B (jlk) - large blocks, truly interleaved
{1, 1, 1}
1
{64, 128, 64}
0, 0, 0
)";

// Keep-last dispatcher tests (A uses [C,F,F,C], rank-4):
// einsum: jikl,jlmn->ikmn
// A modes "jikl": j=contracted(pos0), i=free(pos1), k=free(pos2), l=contracted(pos3)
//   -> interleaved, non-single-swap permutation; generic-permute path expected.
// B modes "jlmn": j,l=contracted then m,n=free -> already normal.
inline const char* GENERIC_PERMUTE_A = R"(
# Tensor A (jikl) - generic-permute vectorizable tail case (l=4)
{1, 1, 1, 1}
1
{2, 8, 1, 4}
0, 0, 0, 0
)";

inline const char* GENERIC_PERMUTE_B = R"(
# Tensor B (jlmn) - generic-permute vectorizable tail case (l=4)
{1, 1, 1, 1}
1
{2, 4, 8, 1}
0, 0, 0, 0
)";

inline const char* GENERIC_PERMUTE_NONVEC_A = R"(
# Tensor A (jikl) - generic-permute scalar fallback case (tail l=2)
{1, 1, 1, 1}
1
{2, 8, 1, 2}
0, 0, 0, 0
)";

inline const char* GENERIC_PERMUTE_NONVEC_B = R"(
# Tensor B (jlmn) - generic-permute scalar fallback case (tail l=2)
{1, 1, 1, 1}
1
{2, 2, 8, 1}
0, 0, 0, 0
)";

inline const char* GENERIC_PERMUTE_MIXED_A = R"(
# Tensor A (jikl) - generic-permute mixed vector/scalar blocks (l sections 4 and 2)
{1, 1, 1, 2}
2
{2, 8, 1, 4, 2}
0, 0, 0, 0
0, 0, 0, 1
)";

inline const char* GENERIC_PERMUTE_MIXED_B = R"(
# Tensor B (jlmn) - generic-permute mixed vector/scalar blocks (l sections 4 and 2)
{1, 2, 1, 1}
2
{2, 4, 2, 8, 1}
0, 0, 0, 0
0, 1, 0, 0
)";

// ===============================================================
// Medium Kernel Tests
// Medium tier: M,N >= 8 (MIN_CUTLASS_DIM) but M*N < 4096.
// These target the shared-memory tiled GEMM kernel.
// ===============================================================

// --- Pure medium NN: ij,jk->ik, M=16, N=16, K=32, M*N=256 ---
inline const char* MED_NN_A = R"(
# Tensor A (ij) - medium block 16x32
{1, 1}
1
{16, 32}
0, 0
)";

inline const char* MED_NN_B = R"(
# Tensor B (jk) - medium block 32x16
{1, 1}
1
{32, 16}
0, 0
)";

// --- Medium TN: ji,jk->ik, M=32, N=24, K=20, M*N=768 ---
inline const char* MED_TN_A = R"(
# Tensor A (ji) - medium TN block
{1, 1}
1
{20, 32}
0, 0
)";

inline const char* MED_TN_B = R"(
# Tensor B (jk) - medium TN block
{1, 1}
1
{20, 24}
0, 0
)";

// --- Medium NT: ij,kj->ik, M=24, N=32, K=20, M*N=768 ---
inline const char* MED_NT_A = R"(
# Tensor A (ij) - medium NT block
{1, 1}
1
{24, 20}
0, 0
)";

inline const char* MED_NT_B = R"(
# Tensor B (kj) - medium NT block
{1, 1}
1
{32, 20}
0, 0
)";

// --- Medium TT: ji,kj->ik, M=32, N=24, K=20, M*N=768 ---
// Reuse MED_TN_A for A(ji) and MED_NT_B for B(kj)

// --- Medium near-limit: M=63, N=63, K=8, M*N=3969 (just under 4096) ---
inline const char* MED_NEAR_LIMIT_A = R"(
# Tensor A (ij) - near threshold
{1, 1}
1
{63, 8}
0, 0
)";

inline const char* MED_NEAR_LIMIT_B = R"(
# Tensor B (jk) - near threshold
{1, 1}
1
{8, 63}
0, 0
)";

// --- Medium wide: M=8, N=64, K=128, M*N=512 (M at minimum, large K) ---
inline const char* MED_WIDE_A = R"(
# Tensor A (ij) - wide M=8, K=128
{1, 1}
1
{8, 128}
0, 0
)";

inline const char* MED_WIDE_B = R"(
# Tensor B (jk) - wide N=64
{1, 1}
1
{128, 64}
0, 0
)";

// --- Medium tall: M=64, N=8, K=128, M*N=512 ---
inline const char* MED_TALL_A = R"(
# Tensor A (ij) - tall M=64, K=128
{1, 1}
1
{64, 128}
0, 0
)";

inline const char* MED_TALL_B = R"(
# Tensor B (jk) - tall N=8
{1, 1}
1
{128, 8}
0, 0
)";

// --- Medium accumulation: 2 pairs map to same output block ---
// A has i=16, j sections {0,1}; B has j sections {0,1}, k=16
// Pair (A[0,0],B[0,0]) and (A[0,1],B[1,0]) both produce output block (0,0)
inline const char* MED_ACCUM_A = R"(
# Tensor A (ij) - medium accum
{1, 2}
2
{16, 20, 20}
0, 0
0, 1
)";

inline const char* MED_ACCUM_B = R"(
# Tensor B (jk) - medium accum
{2, 1}
2
{20, 20, 16}
0, 0
1, 0
)";

// --- Medium batched: bij,bjk->bik, M=16, N=16, K=20, batch=2 ---
inline const char* MED_BATCH_A = R"(
# Tensor A (bij) - medium batched
{2, 1, 1}
2
{1, 1, 16, 20}
0, 0, 0
1, 0, 0
)";

inline const char* MED_BATCH_B = R"(
# Tensor B (bjk) - medium batched
{2, 1, 1}
2
{1, 1, 20, 16}
0, 0, 0
1, 0, 0
)";

// --- Medium 3D: ijk,jkl->il, M=16, N=16, K=5*4=20, M*N=256 ---
inline const char* MED_3D_A = R"(
# Tensor A (ijk) - medium 3D
{1, 1, 1}
1
{16, 5, 4}
0, 0, 0
)";

inline const char* MED_3D_B = R"(
# Tensor B (jkl) - medium 3D
{1, 1, 1}
1
{5, 4, 16}
0, 0, 0
)";

// --- Medium + transpose: jik,jkl->il, A interleaved [C,F,C] ---
// M=16, N=16, K=5*4=20, M*N=256 (medium), requires transpose
inline const char* MED_TRANSPOSE_A = R"(
# Tensor A (jik) - medium, interleaved [C,F,C]
{1, 1, 1}
1
{5, 16, 4}
0, 0, 0
)";

inline const char* MED_TRANSPOSE_B = R"(
# Tensor B (jkl) - medium
{1, 1, 1}
1
{5, 4, 16}
0, 0, 0
)";

// --- Mixed large+medium+small: ij,jk->ik with varied block sizes ---
// A has i-sections: {128, 16, 4, 2}; j-section: {32}
// B has j-section: {32}; k-sections: {128, 16, 4, 2}
// Pairs (selected):
//   (i=128, k=128) → M*N=16384 → CUTLASS large (Square 128x128)
//   (i=128, k=16)  → M*N=2048  → CUTLASS large (Square)
//   (i=128, k=4)   → M*N=512   → CUTLASS skinny (N=4, min≥4)
//   (i=16,  k=16)  → M*N=256   → CUTLASS large (min=16≥16)
//   (i=4,   k=128) → M*N=512   → CUTLASS skinny (M=4, min≥4)
//   (i=4,   k=4)   → M*N=16    → CUTLASS skinny (min=4≥4)
//   (i=2,   k=128) → M*N=256   → warp/small (M=2 < 4)
//   (i=2,   k=2)   → M*N=4     → warp/small (both < 4)
inline const char* MED_THREE_TIER_A = R"(
# Tensor A (ij) - three-tier dispatch test
{4, 1}
4
{128, 16, 4, 2, 32}
0, 0
1, 0
2, 0
3, 0
)";

inline const char* MED_THREE_TIER_B = R"(
# Tensor B (jk) - three-tier dispatch test
{1, 4}
4
{32, 128, 16, 4, 2}
0, 0
0, 1
0, 2
0, 3
)";

// --- Medium multi-block: many medium pairs with accumulation ---
// A has 4 i-sections (all 16) and 4 j-sections (all 10)
// B has 4 j-sections (all 10) and 2 k-sections (both 16)
// Each (i, k) output block gets up to 4 contributing pairs
inline const char* MED_MULTI_A = R"(
# Tensor A (ij) - many medium pairs
{4, 4}
8
{16, 16, 16, 16, 10, 10, 10, 10}
0, 0
0, 2
1, 1
1, 3
2, 0
2, 2
3, 1
3, 3
)";

inline const char* MED_MULTI_B = R"(
# Tensor B (jk) - many medium pairs
{4, 2}
4
{10, 10, 10, 10, 16, 16}
0, 0
1, 1
2, 0
3, 1
)";

// --- Medium double precision ---
// Same shape as MED_NN but validated with double tolerance
// (reuses MED_NN_A/MED_NN_B tensors)

// ===============================================================
// Small Kernel Tests
// Small tier: min(M,N) < 4 → register-only GEMM kernel.
// ===============================================================

// --- Tiny-tiny NN: M=3, N=2, K=4, min(M,N)=2 < 4 ---
inline const char* SMALL_TINY_A = R"(
# Tensor A (ij) - tiny 3x4
{1, 1}
1
{3, 4}
0, 0
)";

inline const char* SMALL_TINY_B = R"(
# Tensor B (jk) - tiny 4x2
{1, 1}
1
{4, 2}
0, 0
)";

// --- Tall-narrow: M=200, N=3, K=16 ---
inline const char* SMALL_TALL_A = R"(
# Tensor A (ij) - tall 200x16
{1, 1}
1
{200, 16}
0, 0
)";

inline const char* SMALL_TALL_B = R"(
# Tensor B (jk) - narrow 16x3
{1, 1}
1
{16, 3}
0, 0
)";

// --- Wide-short: M=3, N=200, K=16 ---
inline const char* SMALL_WIDE_A = R"(
# Tensor A (ij) - short 3x16
{1, 1}
1
{3, 16}
0, 0
)";

inline const char* SMALL_WIDE_B = R"(
# Tensor B (jk) - wide 16x200
{1, 1}
1
{16, 200}
0, 0
)";

// --- Small accumulation: 2 pairs same output block, M=4, N=3, K varies ---
inline const char* SMALL_ACCUM_A = R"(
# Tensor A (ij) - small accum
{1, 2}
2
{4, 5, 6}
0, 0
0, 1
)";

inline const char* SMALL_ACCUM_B = R"(
# Tensor B (jk) - small accum
{2, 1}
2
{5, 6, 3}
0, 0
1, 0
)";

// --- Small batched: bij,bjk->bik, M=3, N=4, K=5, batch=2 ---
inline const char* SMALL_BATCH_A = R"(
# Tensor A (bij) - small batched
{2, 1, 1}
2
{1, 1, 3, 5}
0, 0, 0
1, 0, 0
)";

inline const char* SMALL_BATCH_B = R"(
# Tensor B (bjk) - small batched
{2, 1, 1}
2
{1, 1, 5, 4}
0, 0, 0
1, 0, 0
)";

// --- Small + transpose: jik,jkl->il, A interleaved [C,F,C], M=3, N=4, K=2*3=6 ---
inline const char* SMALL_TRANSPOSE_A = R"(
# Tensor A (jik) - small, interleaved [C,F,C]
{1, 1, 1}
1
{2, 3, 3}
0, 0, 0
)";

inline const char* SMALL_TRANSPOSE_B = R"(
# Tensor B (jkl) - small
{1, 1, 1}
1
{2, 3, 4}
0, 0, 0
)";

// --- Small K=1: degenerate case (outer product slice) ---
inline const char* SMALL_K1_A = R"(
# Tensor A (ij) - K=1
{1, 1}
1
{5, 1}
0, 0
)";

inline const char* SMALL_K1_B = R"(
# Tensor B (jk) - K=1
{1, 1}
1
{1, 3}
0, 0
)";

// --- Small TN: ji,jk->ik, M=4, N=3, K=5 ---
inline const char* SMALL_TN_A = R"(
# Tensor A (ji) - small TN
{1, 1}
1
{5, 4}
0, 0
)";

inline const char* SMALL_TN_B = R"(
# Tensor B (jk) - small TN
{1, 1}
1
{5, 3}
0, 0
)";

// --- Small NT: ij,kj->ik, M=3, N=5, K=4 (using SMALL_TINY_A as A(ij) 3x4) ---
inline const char* SMALL_NT_B = R"(
# Tensor B (kj) - small NT
{1, 1}
1
{5, 4}
0, 0
)";

// --- Four-tier mix: large + medium + small with tiny blocks ---
// Extends MED_THREE_TIER with an additional very small section (M=2)
inline const char* SMALL_FOUR_TIER_A = R"(
# Tensor A (ij) - four-tier dispatch test
{4, 1}
4
{128, 16, 4, 2, 32}
0, 0
1, 0
2, 0
3, 0
)";

inline const char* SMALL_FOUR_TIER_B = R"(
# Tensor B (jk) - four-tier dispatch test
{1, 4}
4
{32, 128, 16, 4, 2}
0, 0
0, 1
0, 2
0, 3
)";

// --- Small interleaved: batch not leading, small-only (no large/medium to trigger transpose) ---
// einsum: ibj,bjk->bik  A(ibj): [F,B,C] interleaved, tiny blocks (M=2, N=3, K=4, batch=2)
inline const char* SMALL_INTERLEAVED_BATCH_A = R"(
# Tensor A (ibj) - small, free/batch/contracted interleaved
{1, 1, 1}
1
{2, 2, 4}
0, 0, 0
)";

inline const char* SMALL_INTERLEAVED_BATCH_B = R"(
# Tensor B (bjk) - small
{1, 1, 1}
1
{2, 4, 3}
0, 0, 0
)";

// --- Small interleaved with dim coalescing ---
// einsum: jkil,jklm->im  A(jkil): [C,C,F,C] → coalesces to [C,F,C], M=3, K=2*4*5=40
// B(jklm): [C,C,C,F] → coalesces to [C,F], N=6, K=2*4*5=40
inline const char* SMALL_INTERLEAVED_COALESCE_A = R"(
# Tensor A (jkil) - small, coalescing test [C,C,F,C]
{1, 1, 1, 1}
1
{2, 4, 3, 5}
0, 0, 0, 0
)";

inline const char* SMALL_INTERLEAVED_COALESCE_B = R"(
# Tensor B (jklm) - small, [C,C,C,F] coalesces to [C,F]
{1, 1, 1, 1}
1
{2, 4, 5, 6}
0, 0, 0, 0
)";

// --- Mixed interleaved: small + large blocks in same contraction ---
// einsum: jik,jkl->il  A(jik): [C,F,C] interleaved
// Two blocks: one small (M=3), one large (M=128). Both share same K structure.
// The large block triggers transpose; small block gets pointer-patched.
inline const char* SMALL_MIXED_INTERLEAVED_A = R"(
# Tensor A (jik) - mixed small+large, interleaved
{1, 2, 1}
2
{4, 3, 128, 8}
0, 0, 0
0, 1, 0
)";

inline const char* SMALL_MIXED_INTERLEAVED_B = R"(
# Tensor B (jkl) - mixed, non-interleaved
{1, 1, 2}
2
{4, 8, 3, 128}
0, 0, 0
0, 0, 1
)";

// --- Small B-only interleaved ---
// einsum: ijk,jlk->il  A(ijk): [F,C,C] OK, B(jlk): [C,F,C] interleaved
// Small blocks: M=3, N=2, K=4*5=20
inline const char* SMALL_B_INTERLEAVED_A = R"(
# Tensor A (ijk) - small, non-interleaved
{1, 1, 1}
1
{3, 4, 5}
0, 0, 0
)";

inline const char* SMALL_B_INTERLEAVED_B = R"(
# Tensor B (jlk) - small, [C,F,C] interleaved
{1, 1, 1}
1
{4, 2, 5}
0, 0, 0
)";

// ===============================================================
// Cross-Cutting Interaction Tests
// These target multi-feature interactions that isolated tests miss.
// ===============================================================

// --- Test 1: GP large mixed trivial — jik,jkl->il ---
inline const char* CROSS_GP_MIXED_TRIVIAL_A = R"(
# Tensor A (jik) - GP mixed trivial
{2, 1, 1}
2
{1, 8, 64, 64}
0, 0, 0
1, 0, 0
)";

inline const char* CROSS_GP_MIXED_TRIVIAL_B = R"(
# Tensor B (jkl) - GP mixed trivial
{2, 1, 1}
2
{1, 8, 64, 64}
0, 0, 0
1, 0, 0
)";

// --- Test 2: Large 3-pair reduce — ij,jk->ik ---
// 3 CUTLASS pairs (M=128, N=64) all → output(0,0) → 3-way reduce_noatomic.
inline const char* CROSS_3PAIR_REDUCE_A = R"(
# Tensor A (ij) - 3 j-sections
{1, 3}
3
{128, 32, 32, 32}
0, 0
0, 1
0, 2
)";

inline const char* CROSS_3PAIR_REDUCE_B = R"(
# Tensor B (jk) - 3 j-sections
{3, 1}
3
{32, 32, 32, 64}
0, 0
1, 0
2, 0
)";

// --- Test 3: Direct-write + multi-pair — ij,jk->ik ---
// Output(0,0): 2 pairs → multi-pair reduce. Output(0,1): 1 pair → direct-write.
inline const char* CROSS_DIRECT_MULTI_A = R"(
# Tensor A (ij) - 3 j-sections
{1, 3}
3
{128, 32, 32, 32}
0, 0
0, 1
0, 2
)";

inline const char* CROSS_DIRECT_MULTI_B = R"(
# Tensor B (jk) - 3 j-sections, 2 k-sections
{3, 2}
3
{32, 32, 32, 64, 64}
0, 0
1, 0
2, 1
)";

// --- Test 4: Mixed volume+CUTLASS output block — ij,jk->ik ---
// Volume pair (M=16,N=16,K=8, vol=256) stays in volume tier;
// CUTLASS pair (M=16,N=16,K=260, vol=8320 > VOL_THRESHOLD_F32) writes to temp buffer.
// Both target the same output block — reduce merges temp→d_output after sync.
inline const char* CROSS_PROMOTION_A = R"(
# Tensor A (ij) - 2 j-sections with different K
{1, 2}
2
{16, 8, 260}
0, 0
0, 1
)";

inline const char* CROSS_PROMOTION_B = R"(
# Tensor B (jk) - 2 j-sections
{2, 1}
2
{8, 260, 16}
0, 0
1, 0
)";

// --- Test 5: Batch=4 CUTLASS — bij,bjk->bik ---
// 4 CUTLASS singletons (M=64, N=64, K=64).
inline const char* CROSS_BATCH4_A = R"(
# Tensor A (bij) - batch=4
{4, 1, 1}
4
{1, 1, 1, 1, 64, 64}
0, 0, 0
1, 0, 0
2, 0, 0
3, 0, 0
)";

inline const char* CROSS_BATCH4_B = R"(
# Tensor B (bjk) - batch=4
{4, 1, 1}
4
{1, 1, 1, 1, 64, 64}
0, 0, 0
1, 0, 0
2, 0, 0
3, 0, 0
)";

// --- Test 6: Batch=4 mixed tiers — bij,bjk->bik ---
// Batch 0,1: M=64,N=64,K=32 → CUTLASS. Batch 2,3: M=4,N=64,K=32 → small (min=4<8).
inline const char* CROSS_BATCH4_MIXED_A = R"(
# Tensor A (bij) - batch=4, mixed i-sections
{4, 2, 1}
4
{1, 1, 1, 1, 64, 4, 32}
0, 0, 0
1, 0, 0
2, 1, 0
3, 1, 0
)";

inline const char* CROSS_BATCH4_MIXED_B = R"(
# Tensor B (bjk) - batch=4
{4, 1, 1}
4
{1, 1, 1, 1, 32, 64}
0, 0, 0
1, 0, 0
2, 0, 0
3, 0, 0
)";

// --- Test 7: Three-tier multi-pair — ij,jk->ik ---
// 3 i-sections (64,16,4), 2 j-sections (16,16), 3 k-sections (64,16,4).
// 9 output blocks × 2 pairs each = 18 pairs across all tiers.
inline const char* CROSS_THREE_TIER_MULTI_A = R"(
# Tensor A (ij) - three-tier multi-pair
{3, 2}
6
{64, 16, 4, 16, 16}
0, 0
0, 1
1, 0
1, 1
2, 0
2, 1
)";

inline const char* CROSS_THREE_TIER_MULTI_B = R"(
# Tensor B (jk) - three-tier multi-pair
{2, 3}
6
{16, 16, 64, 16, 4}
0, 0
0, 1
0, 2
1, 0
1, 1
1, 2
)";

// --- Test 8: Batch=4 GenericPermute — bjik,bjkl->bil ---
// 4 CUTLASS pairs with GenericPermute on A (layout [B,C,F,C]).
inline const char* CROSS_BATCH4_GP_A = R"(
# Tensor A (bjik) - batch=4, GP [B,C,F,C]
{4, 1, 1, 1}
4
{1, 1, 1, 1, 16, 64, 16}
0, 0, 0, 0
1, 0, 0, 0
2, 0, 0, 0
3, 0, 0, 0
)";

inline const char* CROSS_BATCH4_GP_B = R"(
# Tensor B (bjkl) - batch=4
{4, 1, 1, 1}
4
{1, 1, 1, 1, 16, 16, 64}
0, 0, 0, 0
1, 0, 0, 0
2, 0, 0, 0
3, 0, 0, 0
)";

// --- Test 9: GP mixed trivial + large+small — jik,jkl->il ---
// A has 2 j-sections and 2 i-sections (64,4). j=1: GP-trivial, j=8: GP-non-trivial.
// Output(0,0): 2 CUTLASS pairs (multi-pair reduce).
// Output(1,0): 2 small pairs (min(M,N)=4<8, atomic accum).
inline const char* CROSS_GP_TIERS_A = R"(
# Tensor A (jik) - GP mixed trivial across tiers
{2, 2, 1}
4
{1, 8, 64, 4, 64}
0, 0, 0
0, 1, 0
1, 0, 0
1, 1, 0
)";

inline const char* CROSS_GP_TIERS_B = R"(
# Tensor B (jkl) - GP mixed trivial across tiers
{2, 1, 1}
2
{1, 8, 64, 64}
0, 0, 0
1, 0, 0
)";

// ===============================================================
// Skinny CUTLASS Tile Tests
// Blocks with M*N >= 4096 but min(M,N) < 16 trigger SkinnyTall
// (Tile_64x8) or SkinnyWide (Tile_8x64) shape queues.
// ===============================================================

// --- Skinny-tall: ij,jk->ik  M=512, N=8, K=16 → SkinnyTall (Tile_64x8) ---
inline const char* SKINNY_TALL_A = R"(
# Tensor A (ij) - skinny-tall block M=512
{1, 1}
1
{512, 16}
0, 0
)";

inline const char* SKINNY_TALL_B = R"(
# Tensor B (jk) - skinny-tall block N=8
{1, 1}
1
{16, 8}
0, 0
)";

// --- Skinny-wide: ij,jk->ik  M=8, N=512, K=16 → SkinnyWide (Tile_8x64) ---
inline const char* SKINNY_WIDE_A = R"(
# Tensor A (ij) - skinny-wide block M=8
{1, 1}
1
{8, 16}
0, 0
)";

inline const char* SKINNY_WIDE_B = R"(
# Tensor B (jk) - skinny-wide block N=512
{1, 1}
1
{16, 512}
0, 0
)";

// --- Skinny multi-block: 2 blocks with different skinny shapes ---
// Block 0: M=512, N=10, K=16 → SkinnyTall
// Block 1: M=10, N=512, K=16 → SkinnyWide
inline const char* SKINNY_MIXED_A = R"(
# Tensor A (ij) - mixed skinny blocks
{2, 1}
2
{512, 10, 16}
0, 0
1, 0
)";

inline const char* SKINNY_MIXED_B = R"(
# Tensor B (jk) - mixed skinny blocks
{1, 2}
2
{16, 10, 512}
0, 0
0, 1
)";

} // namespace test_tensors
