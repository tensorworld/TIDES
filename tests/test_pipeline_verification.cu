// Pipeline verification test suite.
// Tests execute_general_sparse_pipeline() against cuTENSOR block-sparse reference.
// Also cross-validates against execute_general_sparse_contraction().
//
// Build: make test_pipeline_verification
// Run: ./test_pipeline_verification [--verbose]

#include "bs_parser.hpp"
#include "einsum.hpp"
#include "general_sparse/gpu_sparse_input.hpp"
#include "general_sparse/general_sparse_executor.cuh"
#include "general_sparse/stride_verifier.hpp"
#include "general_sparse/gemm_layout_check.hpp"
#include "general_sparse/random_contraction.hpp"
#include "general_sparse/cutensor_block_sparse.cuh"
#include "test_utils/test_tensors.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace bsptc;

static bool g_verbose = false;
static bool g_f64 = false;
static int g_passed = 0;
static int g_failed = 0;

const char* dispatch_kind_to_string(TransposeDispatchKind kind) {
    switch (kind) {
        case TransposeDispatchKind::None: return "None";
        case TransposeDispatchKind::GenericPermute: return "GenericPermute";
    }
    return "None";
}

template <typename T>
int generic_permute_vec_elems() {
    if (std::is_same<T, float>::value) return 4;
    if (std::is_same<T, double>::value) return 2;
    return 0;
}

template <typename T>
int64_t count_generic_permute_vectorizable_blocks(
    const GPUSparseInput<T>& gpu,
    const GemmLayoutPlan::OperandPlan& plan)
{
    if (!plan.transpose || plan.dispatch_kind != TransposeDispatchKind::GenericPermute) return 0;

    const int vec = generic_permute_vec_elems<T>();
    if (vec <= 0 || plan.unmoved_tail_dims <= 0) return 0;

    const dim_t head_rank = gpu.rank - plan.unmoved_tail_dims;
    int64_t count = 0;
    for (int64_t blk = 0; blk < gpu.num_blocks; ++blk) {
        const dim_t* dims = &gpu.block_dims[blk * gpu.rank];
        int64_t tail_volume = 1;
        for (int32_t d = head_rank; d < gpu.rank; ++d) {
            tail_volume *= dims[d];
        }
        int64_t volume = gpu.block_volumes[blk];
        int64_t offset = gpu.block_offsets[blk];
        bool eligible =
            (tail_volume > 0) &&
            (tail_volume % vec == 0) &&
            (volume % vec == 0) &&
            (offset % vec == 0);
        if (eligible) count++;
    }
    return count;
}

std::string create_temp_tensor(const std::string& content) {
    static int counter = 0;
    std::string filename = "/tmp/test_pipeline_" + std::to_string(counter++) + ".bs";
    std::ofstream file(filename);
    file << content;
    file.close();
    return filename;
}

// Run a pipeline test with stride-based verification
template <typename T>
bool run_pipeline_test(
    const std::string& name,
    const std::string& einsum_str,
    const std::string& tensor_A_content,
    const std::string& tensor_B_content,
    double atol = 1e-5,
    double rtol = 1e-5,
    int64_t mn_threshold = DEFAULT_CUTLASS_MN_THRESHOLD)
{
    std::cout << "Test: " << name << " ... " << std::flush;

    try {
        std::string fileA = create_temp_tensor(tensor_A_content);
        std::string fileB = create_temp_tensor(tensor_B_content);

        DataType dtype = std::is_same<T, double>::value ? DataType::FLOAT64 : DataType::FLOAT32;
        auto tensors = parse_tensor_files({fileA, fileB}, dtype);
        BlockSparseTensor& A = tensors[0];
        BlockSparseTensor& B = tensors[1];

        auto expr = parse_einsum(einsum_str);
        std::vector<std::vector<int64_t>> shapes = {
            block_utils::compute_full_shape(A),
            block_utils::compute_full_shape(B)
        };
        validate_einsum(expr, shapes);

        EinsumIndexSets index_sets;
        create_einsum_index_sets(expr, index_sets);

        auto A_gpu = to_gpu_input<T>(A);
        auto B_gpu = to_gpu_input<T>(B);

        // Build config for stride verifier
        ContractionConfig config = build_contraction_config(
            index_sets, A_gpu.labels, B_gpu.labels, expr.output_labels);

        GemmLayoutPlan layout_plan = plan_gemm_layout(config, A_gpu.rank, B_gpu.rank);
        if (g_verbose) {
            std::cout << "  Dispatch plan: mode=" << gemm_mode_to_string(layout_plan.mode_after_transpose)
                      << " | A=" << dispatch_kind_to_string(layout_plan.plan_A.dispatch_kind)
                      << " (transpose=" << (layout_plan.plan_A.transpose ? "yes" : "no")
                      << ")"
                      << " | B=" << dispatch_kind_to_string(layout_plan.plan_B.dispatch_kind)
                      << " (transpose=" << (layout_plan.plan_B.transpose ? "yes" : "no")
                      << ")\n";
            if (layout_plan.plan_A.dispatch_kind == TransposeDispatchKind::GenericPermute) {
                int64_t vec_ok = count_generic_permute_vectorizable_blocks(A_gpu, layout_plan.plan_A);
                std::cout << "    A generic-permute vec-eligible blocks: "
                          << vec_ok << "/" << A_gpu.num_blocks << "\n";
            }
            if (layout_plan.plan_B.dispatch_kind == TransposeDispatchKind::GenericPermute) {
                int64_t vec_ok = count_generic_permute_vectorizable_blocks(B_gpu, layout_plan.plan_B);
                std::cout << "    B generic-permute vec-eligible blocks: "
                          << vec_ok << "/" << B_gpu.num_blocks << "\n";
            }
        }

        // Run pipeline
        DispatchConfig dc;
        dc.mn_threshold = mn_threshold;
        auto A_dev = copy_sparse_to_device(A_gpu);
        auto B_dev = copy_sparse_to_device(B_gpu);
        auto pipeline_result = execute_general_sparse_pipeline(
            A_gpu, B_gpu, A_dev, B_dev, index_sets, expr.output_labels, dc);

        std::remove(fileA.c_str());
        std::remove(fileB.c_str());

        if (pipeline_result.total_output_elements == 0) {
            std::cout << "PASSED (empty result)\n";
            g_passed++;
            return true;
        }

        // Compute cuTENSOR block-sparse reference
        auto bs_ref = run_cutensor_block_sparse<T>(A, B, expr);

        // Verify pipeline output against cuTENSOR block-sparse
        auto vr = verify_against_block_sparse(
            pipeline_result, A_gpu, B_gpu, config, bs_ref, atol, rtol);
        if (!vr.passed) {
            std::cout << "FAILED (stride max_rel=" << std::scientific << vr.max_rel
                      << " max_abs=" << vr.max_abs
                      << " atol=" << atol << " rtol=" << rtol << ")\n";
            g_failed++;
            return false;
        }

        std::cout << "PASSED (rel=" << std::scientific << std::setprecision(2)
                  << vr.max_rel
                  << ", bs_cutensor=" << std::fixed << std::setprecision(2)
                  << bs_ref.kernel_ms << "ms)\n";

        g_passed++;
        return true;
    } catch (const std::exception& e) {
        std::cout << "FAILED (exception: " << e.what() << ")\n";
        g_failed++;
        return false;
    }
}

// Test that PipelineTimingBreakdown fields are correct for known contractions.
template <typename T>
bool run_timing_stats_test(
    const std::string& name,
    const std::string& einsum_str,
    const std::string& tensor_A_content,
    const std::string& tensor_B_content,
    // Expected tier occupancy (true = tier should have work)
    bool expect_large, bool expect_medium, bool expect_small,
    // Whether transpose should actually run (not just be "needed")
    bool expect_transpose_runs,
    int64_t mn_threshold = DEFAULT_CUTLASS_MN_THRESHOLD)
{
    std::cout << "Test: " << name << " ... " << std::flush;

    try {
        std::string fileA = create_temp_tensor(tensor_A_content);
        std::string fileB = create_temp_tensor(tensor_B_content);

        DataType dtype = std::is_same<T, double>::value ? DataType::FLOAT64 : DataType::FLOAT32;
        auto tensors = parse_tensor_files({fileA, fileB}, dtype);
        BlockSparseTensor& A = tensors[0];
        BlockSparseTensor& B = tensors[1];

        auto expr = parse_einsum(einsum_str);
        std::vector<std::vector<int64_t>> shapes = {
            block_utils::compute_full_shape(A),
            block_utils::compute_full_shape(B)
        };
        validate_einsum(expr, shapes);

        EinsumIndexSets index_sets;
        create_einsum_index_sets(expr, index_sets);

        auto A_gpu = to_gpu_input<T>(A);
        auto B_gpu = to_gpu_input<T>(B);

        DispatchConfig dc;
        dc.mn_threshold = mn_threshold;

        // Run with timing, then reuse to check stale-field bug
        PipelineTimingBreakdown tb;
        auto A_dev = copy_sparse_to_device(A_gpu);
        auto B_dev = copy_sparse_to_device(B_gpu);
        auto result = execute_general_sparse_pipeline<T>(
            A_gpu, B_gpu, A_dev, B_dev, index_sets, expr.output_labels, dc, &tb);

        std::remove(fileA.c_str());
        std::remove(fileB.c_str());

        bool ok = true;
        auto fail = [&](const std::string& msg) {
            if (ok) std::cout << "FAILED\n";
            std::cout << "  STAT FAIL: " << msg << "\n";
            ok = false;
        };

        // Invariant: output elems sum == total_pair_output_size
        // (total_pair_output_size is in the plan, but we can check via result)
        int64_t tier_sum = tb.large_output_elems + tb.medium_output_elems + tb.small_output_elems;
        if (result.total_output_elements > 0 && tier_sum == 0)
            fail("tier output elem sum is 0 but result has elements");

        // Check tier occupancy
        if (expect_large && tb.large_gemm_problems == 0)
            fail("expected large problems but got 0");
        if (!expect_large && tb.large_gemm_problems != 0)
            fail("expected no large problems but got " + std::to_string(tb.large_gemm_problems));
        if (expect_medium && tb.medium_descs == 0)
            fail("expected medium descs but got 0");
        if (!expect_medium && tb.medium_descs != 0)
            fail("expected no medium descs but got " + std::to_string(tb.medium_descs));
        if (expect_small && tb.small_descs == 0)
            fail("expected small descs but got 0");
        if (!expect_small && tb.small_descs != 0)
            fail("expected no small descs but got " + std::to_string(tb.small_descs));

        // FLOPs should be positive for occupied tiers
        if (expect_large && tb.large_flops <= 0)
            fail("large tier occupied but flops=" + std::to_string(tb.large_flops));
        if (expect_medium && tb.medium_flops <= 0)
            fail("medium tier occupied but flops=" + std::to_string(tb.medium_flops));
        if (expect_small && tb.small_flops <= 0)
            fail("small tier occupied but flops=" + std::to_string(tb.small_flops));
        if (!expect_large && tb.large_flops != 0)
            fail("no large work but flops=" + std::to_string(tb.large_flops));
        if (!expect_medium && tb.medium_flops != 0)
            fail("no medium work but flops=" + std::to_string(tb.medium_flops));
        if (!expect_small && tb.small_flops != 0)
            fail("no small work but flops=" + std::to_string(tb.small_flops));

        // small_descs == 0 => small_kernel_kind == nullptr
        if (tb.small_descs == 0 && tb.small_kernel_kind != nullptr)
            fail("small_descs=0 but small_kernel_kind is set");
        if (tb.small_descs > 0 && tb.small_kernel_kind == nullptr)
            fail("small_descs>0 but small_kernel_kind is null");

        // Transpose: if not running, bytes and kinds should be zero/null
        if (!expect_transpose_runs) {
            if (tb.transpose_bytes != 0)
                fail("transpose not expected to run but bytes=" + std::to_string(tb.transpose_bytes));
            if (tb.transpose_A_kind != nullptr)
                fail("transpose not expected to run but A_kind set");
            if (tb.transpose_B_kind != nullptr)
                fail("transpose not expected to run but B_kind set");
        }
        // If transpose runs, bytes should be positive
        if (expect_transpose_runs && tb.transpose_bytes <= 0)
            fail("transpose expected to run but bytes=" + std::to_string(tb.transpose_bytes));

        // transpose_bytes > 0 => at least one kind set
        if (tb.transpose_bytes > 0 && tb.transpose_A_kind == nullptr && tb.transpose_B_kind == nullptr)
            fail("transpose_bytes>0 but no kind set");

        if (ok) {
            std::cout << "PASSED\n";
            g_passed++;
        } else {
            g_failed++;
        }
        return ok;
    } catch (const std::exception& e) {
        std::cout << "FAILED (exception: " << e.what() << ")\n";
        g_failed++;
        return false;
    }
}

// Run a pipeline test from a fuzzer seed (reproduces random contraction failures).
// Uses the same randomize_params + generate_random_contraction flow as fuzz_pipeline.
template <typename T>
bool run_fuzz_seed_test(
    const std::string& name,
    uint64_t seed,
    int max_rank = 5,
    int max_sections = 5,
    int max_section_size = 64,
    double density = 0.3,
    double atol = 1e-5,
    double rtol = 1e-5,
    int64_t mn_threshold = DEFAULT_CUTLASS_MN_THRESHOLD)
{
    std::cout << "Test: " << name << " ... " << std::flush;

    try {
        std::mt19937 meta_rng(seed);
        auto params = randomize_params(meta_rng, max_rank, max_sections,
                                       max_section_size, density, seed);

        DataType dtype = std::is_same<T, double>::value ? DataType::FLOAT64 : DataType::FLOAT32;
        auto rc = generate_random_contraction(params, dtype);

        auto expr = parse_einsum(rc.einsum_str);
        std::vector<std::vector<int64_t>> shapes = {
            block_utils::compute_full_shape(rc.A),
            block_utils::compute_full_shape(rc.B)
        };
        validate_einsum(expr, shapes);

        EinsumIndexSets index_sets;
        create_einsum_index_sets(expr, index_sets);

        auto A_gpu = to_gpu_input<T>(rc.A);
        auto B_gpu = to_gpu_input<T>(rc.B);

        ContractionConfig config = build_contraction_config(
            index_sets, A_gpu.labels, B_gpu.labels, expr.output_labels);

        if (g_verbose) {
            GemmLayoutPlan layout_plan = plan_gemm_layout(config, A_gpu.rank, B_gpu.rank);
            std::cout << "\n  einsum=" << rc.einsum_str
                      << " mode=" << gemm_mode_to_string(layout_plan.mode_after_transpose)
                      << " A=" << dispatch_kind_to_string(layout_plan.plan_A.dispatch_kind)
                      << "(t=" << (layout_plan.plan_A.transpose ? "y" : "n") << ")"
                      << " B=" << dispatch_kind_to_string(layout_plan.plan_B.dispatch_kind)
                      << "(t=" << (layout_plan.plan_B.transpose ? "y" : "n") << ")"
                      << "\n  ";
        }

        // Run pipeline
        DispatchConfig dc;
        dc.mn_threshold = mn_threshold;
        auto A_dev = copy_sparse_to_device(A_gpu);
        auto B_dev = copy_sparse_to_device(B_gpu);
        auto pipeline_result = execute_general_sparse_pipeline(
            A_gpu, B_gpu, A_dev, B_dev, index_sets, expr.output_labels, dc);

        if (pipeline_result.total_output_elements == 0) {
            std::cout << "PASSED (empty result)\n";
            g_passed++;
            return true;
        }

        // Compute cuTENSOR block-sparse reference
        auto bs_ref = run_cutensor_block_sparse<T>(rc.A, rc.B, expr);

        // Verify pipeline output against cuTENSOR block-sparse
        auto vr = verify_against_block_sparse(
            pipeline_result, A_gpu, B_gpu, config, bs_ref, atol, rtol);

        if (!vr.passed) {
            std::cout << "FAILED (einsum=" << rc.einsum_str
                      << " max_rel=" << std::scientific << vr.max_rel
                      << " max_abs=" << vr.max_abs
                      << " atol=" << atol << " rtol=" << rtol << ")\n";
            g_failed++;
            return false;
        }

        std::cout << "PASSED (einsum=" << rc.einsum_str
                  << " rel=" << std::scientific << std::setprecision(2)
                  << vr.max_rel
                  << " bs_cutensor=" << std::fixed << std::setprecision(2)
                  << bs_ref.kernel_ms << "ms)\n";

        g_passed++;
        return true;
    } catch (const std::exception& e) {
        std::cout << "FAILED (exception: " << e.what() << ")\n";
        g_failed++;
        return false;
    }
}

using namespace test_tensors;

// Representative fp64 subset covering all tiers and code paths.
// Uses tighter tolerance (1e-10) appropriate for double precision.
static void run_f64_subset() {
    constexpr double D_ATOL = 1e-10;
    constexpr double D_RTOL = 1e-10;

    std::cout << "\n--- FP64 Core Tests ---\n";
    run_pipeline_test<double>("NN matrix ij,jk->ik", "ij,jk->ik", MAT_A, MAT_B, D_ATOL, D_RTOL);
    run_pipeline_test<double>("Accumulation", "ij,jk->ik", ACCUM_A, ACCUM_B, D_ATOL, D_RTOL);
    run_pipeline_test<double>("Transpose output ij,jk->ki", "ij,jk->ki", MAT_A, MAT_B, D_ATOL, D_RTOL);

    std::cout << "\n--- FP64 Rank Tests ---\n";
    run_pipeline_test<double>("3D double contraction ijk,jkl->il", "ijk,jkl->il", T3D_A, T3D_B, D_ATOL, D_RTOL);
    run_pipeline_test<double>("4D GEMM-like abcd,efcd->abef", "abcd,efcd->abef", T4D_A, T4D_B, D_ATOL, D_RTOL);

    std::cout << "\n--- FP64 GEMM Layout Tests (small) ---\n";
    run_pipeline_test<double>("TN matrix ji,jk->ik", "ji,jk->ik", TN_MAT_A, MAT_B, D_ATOL, D_RTOL);
    run_pipeline_test<double>("NT matrix ij,kj->ik", "ij,kj->ik", MAT_A, NT_MAT_B, D_ATOL, D_RTOL);
    run_pipeline_test<double>("TT matrix ji,kj->ik", "ji,kj->ik", TN_MAT_A, NT_MAT_B, D_ATOL, D_RTOL);

    std::cout << "\n--- FP64 GEMM Layout Tests (large, CUTLASS) ---\n";
    run_pipeline_test<double>("Large TN matrix ji,jk->ik", "ji,jk->ik", LARGE_TN_MAT_A, LARGE_TN_MAT_B, D_ATOL, D_RTOL);
    run_pipeline_test<double>("Large NT matrix ij,kj->ik", "ij,kj->ik", LARGE_NT_MAT_A, LARGE_NT_MAT_B, D_ATOL, D_RTOL);

    std::cout << "\n--- FP64 Skinny CUTLASS Tile Tests (64x8 / 8x64) ---\n";
    run_pipeline_test<double>("Skinny-tall M=512 N=8 (Tile_64x8)", "ij,jk->ik",
        SKINNY_TALL_A, SKINNY_TALL_B, D_ATOL, D_RTOL);
    run_pipeline_test<double>("Skinny-wide M=8 N=512 (Tile_8x64)", "ij,jk->ik",
        SKINNY_WIDE_A, SKINNY_WIDE_B, D_ATOL, D_RTOL);
    run_pipeline_test<double>("Skinny mixed tall+wide", "ij,jk->ik",
        SKINNY_MIXED_A, SKINNY_MIXED_B, D_ATOL, D_RTOL);

    std::cout << "\n--- FP64 Must-Transpose Tests ---\n";
    run_pipeline_test<double>("A interleaved [C,F,C] (jik,jkl->il)", "jik,jkl->il",
        INTERLEAVED_A_A, INTERLEAVED_A_B, D_ATOL, D_RTOL);
    run_pipeline_test<double>("B interleaved [C,F,C] (ijk,jlk->il)", "ijk,jlk->il",
        INTERLEAVED_B_A, INTERLEAVED_B_TENSOR, D_ATOL, D_RTOL);

    std::cout << "\n--- FP64 Medium Kernel Tests ---\n";
    run_pipeline_test<double>("Medium NN 16x16 K=32", "ij,jk->ik", MED_NN_A, MED_NN_B, D_ATOL, D_RTOL);
    run_pipeline_test<double>("Medium accumulation", "ij,jk->ik", MED_ACCUM_A, MED_ACCUM_B, D_ATOL, D_RTOL);

    std::cout << "\n--- FP64 Small Kernel Tests ---\n";
    run_pipeline_test<double>("Small tiny 3x2 K=4", "ij,jk->ik", SMALL_TINY_A, SMALL_TINY_B, D_ATOL, D_RTOL);
    run_pipeline_test<double>("Small accumulation", "ij,jk->ik", SMALL_ACCUM_A, SMALL_ACCUM_B, D_ATOL, D_RTOL);

    std::cout << "\n--- FP64 Small Interleaved Tests ---\n";
    run_pipeline_test<double>("Small interleaved batch ibj,bjk->bik", "ibj,bjk->bik",
        SMALL_INTERLEAVED_BATCH_A, SMALL_INTERLEAVED_BATCH_B, D_ATOL, D_RTOL);

    std::cout << "\n--- FP64 Mixed Dispatch Tests ---\n";
    // Note: MIXED_DISPATCH test skipped in fp64 — known CUTLASS grouped GEMM
    // issue with certain mixed-tier block sizes in double precision.
    run_pipeline_test<double>("Three-tier large+medium+small", "ij,jk->ik",
        MED_THREE_TIER_A, MED_THREE_TIER_B, 1e-8, 1e-8);
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v") g_verbose = true;
        else if (arg == "--dtype" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "f64" || v == "float64") g_f64 = true;
        }
    }

    std::cout << "========================================\n";
    std::cout << "Pipeline Verification Test Suite\n";
    std::cout << "========================================\n\n";

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "GPU: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")\n\n";

    if (g_f64) {
        std::cout << "Running fp64 representative subset\n";
        run_f64_subset();

        std::cout << "\n========================================\n";
        std::cout << "Results: " << g_passed << " passed, " << g_failed << " failed (fp64 subset)\n";
        std::cout << "========================================\n";
        return g_failed > 0 ? 1 : 0;
    }

    // Core tests
    std::cout << "--- Core Pipeline Tests (NN layout) ---\n";
    run_pipeline_test<float>("NN matrix ij,jk->ik", "ij,jk->ik", MAT_A, MAT_B);
    run_pipeline_test<float>("Accumulation", "ij,jk->ik", ACCUM_A, ACCUM_B);
    run_pipeline_test<float>("Transpose output ij,jk->ki", "ij,jk->ki", MAT_A, MAT_B);

    std::cout << "\n--- Tensor Rank Tests ---\n";
    run_pipeline_test<float>("3D double contraction ijk,jkl->il", "ijk,jkl->il", T3D_A, T3D_B);
    run_pipeline_test<float>("3D single contraction ijk,jkl->ikl", "ijk,jkl->ikl", T3D_A, T3D_B);
    run_pipeline_test<float>("4D GEMM-like abcd,efcd->abef", "abcd,efcd->abef", T4D_A, T4D_B);

    std::cout << "\n--- Special Cases ---\n";
    run_pipeline_test<float>("Outer product ij,kl->ijkl", "ij,kl->ijkl", OUTER_A, OUTER_B);
    run_pipeline_test<float>("Batch dimension bij,bjk->bik", "bij,bjk->bik", BATCH_A, BATCH_B);
    run_pipeline_test<float>("Multi-block sparse", "ij,jk->ik", MULTI_A, MULTI_B);

    // GEMM layout variant tests: TN/NT/TT
    // TN = A has [contracted..., free...], NT = B has [free..., contracted...]
    std::cout << "\n--- GEMM Layout Tests: TN/NT/TT (small blocks) ---\n";
    run_pipeline_test<float>("TN matrix ji,jk->ik", "ji,jk->ik", TN_MAT_A, MAT_B);
    run_pipeline_test<float>("NT matrix ij,kj->ik", "ij,kj->ik", MAT_A, NT_MAT_B);
    run_pipeline_test<float>("TT matrix ji,kj->ik", "ji,kj->ik", TN_MAT_A, NT_MAT_B);
    run_pipeline_test<float>("TN batched bji,bjk->bik", "bji,bjk->bik", BATCH_TN_A, BATCH_B);
    run_pipeline_test<float>("NT batched bij,bkj->bik", "bij,bkj->bik", BATCH_A, BATCH_NT_B);
    run_pipeline_test<float>("TT batched bji,bkj->bik", "bji,bkj->bik", BATCH_TN_A, BATCH_NT_B);
    run_pipeline_test<float>("TN 3D jki,jkl->il", "jki,jkl->il", TN_3D_A, TN_3D_B);
    run_pipeline_test<float>("NT 3D ijk,ljk->il", "ijk,ljk->il", INTERLEAVED_B_A, INTERLEAVED_B_B);
    run_pipeline_test<float>("TT 3D jki,ljk->il", "jki,ljk->il", TN_3D_A, INTERLEAVED_B_B);

    std::cout << "\n--- GEMM Layout Tests: TN/NT/TT (large blocks, CUTLASS path) ---\n";
    run_pipeline_test<float>("Large TN matrix ji,jk->ik", "ji,jk->ik", LARGE_TN_MAT_A, LARGE_TN_MAT_B);
    run_pipeline_test<float>("Large NT matrix ij,kj->ik", "ij,kj->ik", LARGE_NT_MAT_A, LARGE_NT_MAT_B);
    run_pipeline_test<float>("Large TT matrix ji,kj->ik", "ji,kj->ik", LARGE_TN_MAT_A, LARGE_NT_MAT_B);
    run_pipeline_test<float>("Large TN 3D jki,jkl->il", "jki,jkl->il", LARGE_TN_3D_A, LARGE_TN_3D_B);
    run_pipeline_test<float>("Large NT 3D ijk,ljk->il", "ijk,ljk->il",
        LARGE_INTERLEAVED_B_A, LARGE_INTERLEAVED_B_B);
    run_pipeline_test<float>("Large TT 3D jki,ljk->il", "jki,ljk->il",
        LARGE_TN_3D_A, LARGE_INTERLEAVED_B_B);

    std::cout << "\n--- Skinny CUTLASS Tile Tests (64x8 / 8x64) ---\n";
    run_pipeline_test<float>("Skinny-tall M=512 N=8 (Tile_64x8)", "ij,jk->ik",
        SKINNY_TALL_A, SKINNY_TALL_B);
    run_pipeline_test<float>("Skinny-wide M=8 N=512 (Tile_8x64)", "ij,jk->ik",
        SKINNY_WIDE_A, SKINNY_WIDE_B);
    run_pipeline_test<float>("Skinny mixed tall+wide", "ij,jk->ik",
        SKINNY_MIXED_A, SKINNY_MIXED_B);

    // Must-transpose tests: truly interleaved dimension groups [C,F,C] that
    // cannot be described by a simple transpose flag — require physical transpose.
    std::cout << "\n--- Must-Transpose Tests (truly interleaved dims) ---\n";
    run_pipeline_test<float>(
        "A interleaved [C,F,C] (jik,jkl->il)", "jik,jkl->il",
        INTERLEAVED_A_A, INTERLEAVED_A_B);
    run_pipeline_test<float>(
        "Batch not leading [F,B,C] (ibj,bjk->bik)", "ibj,bjk->bik",
        BATCH_NONLEAD_A, BATCH_NONLEAD_B);
    run_pipeline_test<float>(
        "B interleaved [C,F,C] (ijk,jlk->il)", "ijk,jlk->il",
        INTERLEAVED_B_A, INTERLEAVED_B_TENSOR);
    run_pipeline_test<float>(
        "Both interleaved [C,F,C] (jik,jlk->il)", "jik,jlk->il",
        INTERLEAVED_A_A, INTERLEAVED_B_TENSOR);

    std::cout << "\n--- Large-Block Must-Transpose Tests ---\n";
    run_pipeline_test<float>(
        "Large A interleaved (jik,jkl->il)", "jik,jkl->il",
        LARGE_INTERLEAVED_A_A, LARGE_INTERLEAVED_A_B);
    run_pipeline_test<float>(
        "Large batch not leading (ibj,bjk->bik)", "ibj,bjk->bik",
        LARGE_BATCH_NONLEAD_A, LARGE_BATCH_NONLEAD_B);
    run_pipeline_test<float>(
        "Large B interleaved (ijk,jlk->il)", "ijk,jlk->il",
        LARGE_INTERLEAVED_B_A, LARGE_INTERLEAVED_B_TENSOR);
    run_pipeline_test<float>(
        "Large both interleaved (jik,jlk->il)", "jik,jlk->il",
        LARGE_INTERLEAVED_A_A, LARGE_INTERLEAVED_B_TENSOR);
    run_pipeline_test<float>(
        "Keep-last vec-tail4 (jikl,jlmn->ikmn)", "jikl,jlmn->ikmn",
        GENERIC_PERMUTE_A, GENERIC_PERMUTE_B, 1e-5, 64);
    run_pipeline_test<float>(
        "Keep-last nonvec-tail2 (jikl,jlmn->ikmn)", "jikl,jlmn->ikmn",
        GENERIC_PERMUTE_NONVEC_A, GENERIC_PERMUTE_NONVEC_B, 1e-5, 64);
    run_pipeline_test<float>(
        "Keep-last mixed-tail (jikl,jlmn->ikmn)", "jikl,jlmn->ikmn",
        GENERIC_PERMUTE_MIXED_A, GENERIC_PERMUTE_MIXED_B, 1e-5, 64);

    std::cout << "\n--- Mixed Dispatch Tests (CUTLASS + naive in same run) ---\n";
    run_pipeline_test<float>(
        "Mixed dispatch (ij,jk->ik)", "ij,jk->ik",
        MIXED_DISPATCH_A, MIXED_DISPATCH_B, 1e-4);

    // Medium kernel tests: M,N >= 8 but M*N < 4096 → shared-memory tiled GEMM
    std::cout << "\n--- Medium Kernel Tests ---\n";
    run_pipeline_test<float>("Medium NN 16x16 K=32", "ij,jk->ik", MED_NN_A, MED_NN_B);
    run_pipeline_test<float>("Medium TN 32x24 K=20", "ji,jk->ik", MED_TN_A, MED_TN_B);
    run_pipeline_test<float>("Medium NT 24x32 K=20", "ij,kj->ik", MED_NT_A, MED_NT_B);
    run_pipeline_test<float>("Medium TT 32x24 K=20", "ji,kj->ik", MED_TN_A, MED_NT_B);
    run_pipeline_test<float>("Medium near-limit 63x63 K=8", "ij,jk->ik",
        MED_NEAR_LIMIT_A, MED_NEAR_LIMIT_B);
    run_pipeline_test<float>("Medium wide M=8 N=64 K=128", "ij,jk->ik", MED_WIDE_A, MED_WIDE_B);
    run_pipeline_test<float>("Medium tall M=64 N=8 K=128", "ij,jk->ik", MED_TALL_A, MED_TALL_B);
    run_pipeline_test<float>("Medium accumulation", "ij,jk->ik", MED_ACCUM_A, MED_ACCUM_B);
    run_pipeline_test<float>("Medium batched bij,bjk->bik", "bij,bjk->bik",
        MED_BATCH_A, MED_BATCH_B);
    run_pipeline_test<float>("Medium 3D ijk,jkl->il", "ijk,jkl->il", MED_3D_A, MED_3D_B);
    run_pipeline_test<float>("Medium + transpose jik,jkl->il", "jik,jkl->il",
        MED_TRANSPOSE_A, MED_TRANSPOSE_B);
    run_pipeline_test<float>("Three-tier large+medium+small", "ij,jk->ik",
        MED_THREE_TIER_A, MED_THREE_TIER_B, 1e-4);
    run_pipeline_test<float>("Medium multi-block accumulation", "ij,jk->ik",
        MED_MULTI_A, MED_MULTI_B);
    run_pipeline_test<double>("Medium double precision", "ij,jk->ik", MED_NN_A, MED_NN_B, 1e-10);

    // Small kernel tests: min(M,N) < 8 → register-only GEMM
    std::cout << "\n--- Small Kernel Tests ---\n";
    run_pipeline_test<float>("Small tiny 3x2 K=4", "ij,jk->ik", SMALL_TINY_A, SMALL_TINY_B);
    run_pipeline_test<float>("Small tall 200x3 K=16", "ij,jk->ik", SMALL_TALL_A, SMALL_TALL_B);
    run_pipeline_test<float>("Small wide 3x200 K=16", "ij,jk->ik", SMALL_WIDE_A, SMALL_WIDE_B);
    run_pipeline_test<float>("Small accumulation", "ij,jk->ik", SMALL_ACCUM_A, SMALL_ACCUM_B);
    run_pipeline_test<float>("Small batched bij,bjk->bik", "bij,bjk->bik",
        SMALL_BATCH_A, SMALL_BATCH_B);
    run_pipeline_test<float>("Small + transpose jik,jkl->il", "jik,jkl->il",
        SMALL_TRANSPOSE_A, SMALL_TRANSPOSE_B);
    run_pipeline_test<float>("Small K=1", "ij,jk->ik", SMALL_K1_A, SMALL_K1_B);
    run_pipeline_test<float>("Small TN ji,jk->ik", "ji,jk->ik", SMALL_TN_A, SMALL_TN_B);
    run_pipeline_test<float>("Small NT ij,kj->ik", "ij,kj->ik", SMALL_TINY_A, SMALL_NT_B);
    run_pipeline_test<float>("Four-tier large+medium+small+tiny", "ij,jk->ik",
        SMALL_FOUR_TIER_A, SMALL_FOUR_TIER_B, 1e-4);
    run_pipeline_test<double>("Small double precision", "ij,jk->ik", SMALL_TINY_A, SMALL_TINY_B, 1e-10);

    // Small interleaved tests: small-only blocks with interleaved layouts
    // (no large/medium blocks to trigger physical transpose)
    std::cout << "\n--- Small Interleaved Tests (no-transpose path) ---\n";
    run_pipeline_test<float>("Small interleaved batch ibj,bjk->bik", "ibj,bjk->bik",
        SMALL_INTERLEAVED_BATCH_A, SMALL_INTERLEAVED_BATCH_B);
    run_pipeline_test<float>("Small interleaved coalesce jkil,jklm->im", "jkil,jklm->im",
        SMALL_INTERLEAVED_COALESCE_A, SMALL_INTERLEAVED_COALESCE_B);
    run_pipeline_test<float>("Small B-only interleaved ijk,jlk->il", "ijk,jlk->il",
        SMALL_B_INTERLEAVED_A, SMALL_B_INTERLEAVED_B);
    run_pipeline_test<float>("Small+large interleaved mix jik,jkl->il", "jik,jkl->il",
        SMALL_MIXED_INTERLEAVED_A, SMALL_MIXED_INTERLEAVED_B, 1e-4);

    std::cout << "\n--- Precision Tests ---\n";
    run_pipeline_test<double>("Double precision", "ij,jk->ik", MAT_A, MAT_B, 1e-10);

    std::cout << "\n--- Timing Stats Tests ---\n";
    // Large-only (LARGE_TN uses 128x128 blocks, M*N=16384 >= 4096)
    run_timing_stats_test<float>("Stats: large-only",
        "ji,jk->ik", LARGE_TN_MAT_A, LARGE_TN_MAT_B,
        /*large=*/true, /*medium=*/false, /*small=*/false, /*transpose=*/false);
    // Medium-only (16x16, M*N=256)
    run_timing_stats_test<float>("Stats: medium-only",
        "ij,jk->ik", MED_NN_A, MED_NN_B,
        /*large=*/false, /*medium=*/true, /*small=*/false, /*transpose=*/false);
    // Small-only (3x2, min(M,N)=2 < 8)
    run_timing_stats_test<float>("Stats: small-only",
        "ij,jk->ik", SMALL_TINY_A, SMALL_TINY_B,
        /*large=*/false, /*medium=*/false, /*small=*/true, /*transpose=*/false);
    // Mixed three-tier
    run_timing_stats_test<float>("Stats: three-tier",
        "ij,jk->ik", MED_THREE_TIER_A, MED_THREE_TIER_B,
        /*large=*/true, /*medium=*/true, /*small=*/true, /*transpose=*/false);
    // Large TN 3D — CUTLASS handles TN natively, no physical transpose needed
    run_timing_stats_test<float>("Stats: large TN (no physical transpose)",
        "jki,jkl->il", LARGE_TN_3D_A, LARGE_TN_3D_B,
        /*large=*/true, /*medium=*/false, /*small=*/false, /*transpose=*/false);
    // Medium + interleaved: hypertile GEMM handles interleaved natively, no transpose
    run_timing_stats_test<float>("Stats: medium with interleaved (hypertile, no transpose)",
        "jik,jkl->il", MED_TRANSPOSE_A, MED_TRANSPOSE_B,
        /*large=*/false, /*medium=*/true, /*small=*/false, /*transpose=*/false);

    // Fuzzer regression tests: seeds that found bugs in random contraction testing.
    // These caught: swap-transpose trivial check bug, batch dim ordering bug.
    // FP32 tolerance: atol=1e-3, rtol=1e-3 (accumulation over large K → abs errors ~4e-4).
    constexpr double FUZZ_ATOL = 1e-3;
    constexpr double FUZZ_RTOL = 1e-3;

    std::cout << "\n--- Fuzzer Regression Tests (trivial + batch order bugs) ---\n";
    run_fuzz_seed_test<float>("Fuzz: ebad,adec->abc (2 contracted)",
        1771557395619878558ULL, 5, 5, 64, 0.3, FUZZ_ATOL, FUZZ_RTOL);
    run_fuzz_seed_test<float>("Fuzz: ceab,eda->abcd (1 contracted, diff rank)",
        1771557395619878564ULL, 5, 5, 64, 0.3, FUZZ_ATOL, FUZZ_RTOL);
    run_fuzz_seed_test<float>("Fuzz: bcae,abde->abcd (1 contracted, 2 batch)",
        1771557395619878670ULL, 5, 5, 64, 0.3, FUZZ_ATOL, FUZZ_RTOL);
    run_fuzz_seed_test<float>("Fuzz: bcea,abed->abcd (1 contracted, 2 batch)",
        1771557395619878681ULL, 5, 5, 64, 0.3, FUZZ_ATOL, FUZZ_RTOL);

    std::cout << "\n--- Fuzzer Regression Tests (multi-contracted FP32 precision) ---\n";
    run_fuzz_seed_test<float>("Fuzz: abde,cde->abc (2 contracted)",
        1771557395619878587ULL, 5, 5, 64, 0.3, FUZZ_ATOL, FUZZ_RTOL);
    run_fuzz_seed_test<float>("Fuzz: aedb,deac->abc (2 contracted)",
        1771557395619878613ULL, 5, 5, 64, 0.3, FUZZ_ATOL, FUZZ_RTOL);
    run_fuzz_seed_test<float>("Fuzz: adeb,edca->abc (2 contracted)",
        1771557395619878618ULL, 5, 5, 64, 0.3, FUZZ_ATOL, FUZZ_RTOL);
    run_fuzz_seed_test<float>("Fuzz: aebd,dcae->abc (2 contracted)",
        1771557395619878633ULL, 5, 5, 64, 0.3, FUZZ_ATOL, FUZZ_RTOL);
    run_fuzz_seed_test<float>("Fuzz: bead,ecad->abc (2 contracted)",
        1771557395619878646ULL, 5, 5, 64, 0.3, FUZZ_ATOL, FUZZ_RTOL);
    run_fuzz_seed_test<float>("Fuzz: dcae,cedb->ab (2 contracted)",
        1771557395619878654ULL, 5, 5, 64, 0.3, FUZZ_ATOL, FUZZ_RTOL);
    run_fuzz_seed_test<float>("Fuzz: bade,ceda->abc (2 contracted)",
        1771557395619878728ULL, 5, 5, 64, 0.3, FUZZ_ATOL, FUZZ_RTOL);

    // --- Helper function unit tests ---
    std::cout << "\n--- Helper Function Unit Tests ---\n";

    // Test build_reduction_plan + launch_reduce_noatomic
    // Two medium pairs to same output block → reduction required (not direct-write)
    {
        const char* name = "build_reduction_plan + launch_reduce_noatomic";
        std::cout << "Test: " << name << " ... ";

        BlockPairInfo pairs[2]{};
        pairs[0].output_offset = 0;
        pairs[0].output_volume = 4;
        pairs[1].output_offset = 4;
        pairs[1].output_volume = 4;

        AccumulationMap accum{};
        accum.num_pairs = 2;
        accum.output_rank = 1;
        accum.num_output_blocks = 1;
        accum.total_output_elements = 4;
        int64_t dest_offs[] = {0, 0};  // both write to same output location
        accum.h_dest_offsets = dest_offs;
        int64_t ob_idx[] = {0, 0};
        accum.h_pair_output_block_idx = ob_idx;
        int64_t ob_offset = 0;
        accum.h_output_block_offsets = &ob_offset;
        int64_t ob_volume = 4;
        accum.h_output_block_volumes = &ob_volume;
        int64_t op_offsets[] = {0, 2};  // 1 output block, 2 pairs
        accum.h_output_pair_offsets = op_offsets;
        int64_t op_indices[] = {0, 1};
        accum.h_output_pair_indices = op_indices;

        std::vector<uint8_t> pair_tier = {1, 1}; // both medium tier
        auto rplan = build_reduction_plan(accum, pair_tier, pairs);
        upload_reduction_plan(rplan);
        cudaDeviceSynchronize();

        // temp buffer: pair0=[1,2,3,4], pair1=[10,20,30,40]
        float h_temp[] = {1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f, 40.0f};
        float* d_temp; cudaMalloc(&d_temp, 8 * sizeof(float));
        cudaMemcpy(d_temp, h_temp, 8 * sizeof(float), cudaMemcpyHostToDevice);
        float* d_out; cudaMalloc(&d_out, 4 * sizeof(float));
        cudaMemset(d_out, 0, 4 * sizeof(float));

        launch_reduce_noatomic<float, 256, 4>(d_temp, d_out, rplan);
        cudaDeviceSynchronize();

        float h_out[4];
        cudaMemcpy(h_out, d_out, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_temp); cudaFree(d_out);
        // Don't call free_reduction_plan — it would double-free the stack pointers
        // Just free GPU resources
        if (rplan.d_reduce_dest_offsets) cudaFree(rplan.d_reduce_dest_offsets);
        if (rplan.d_reduce_volumes) cudaFree(rplan.d_reduce_volumes);
        if (rplan.d_reduce_src_offsets) cudaFree(rplan.d_reduce_src_offsets);
        if (rplan.d_reduce_src_ptrs) cudaFree(rplan.d_reduce_src_ptrs);
        delete[] rplan.h_direct_write;

        // Expected: element-wise sum [11, 22, 33, 44]
        bool ok = (h_out[0] == 11.0f && h_out[1] == 22.0f &&
                   h_out[2] == 33.0f && h_out[3] == 44.0f);
        if (ok) {
            std::cout << "PASSED\n"; g_passed++;
        } else {
            std::cout << "FAILED (got " << h_out[0] << "," << h_out[1]
                      << "," << h_out[2] << "," << h_out[3] << ")\n";
            g_failed++;
        }
    }

    // --- Cross-cutting interaction tests ---
    // These target multi-feature interactions that isolated tests miss.
    std::cout << "\n--- Cross-Cutting Interaction Tests ---\n";

    // Test 1: GP large mixed trivial — GenericPermute + CUTLASS + trivial/non-trivial mix
    run_pipeline_test<float>("GP mixed trivial (jik,jkl->il)", "jik,jkl->il",
        CROSS_GP_MIXED_TRIVIAL_A, CROSS_GP_MIXED_TRIVIAL_B, 1e-3);

    // Test 2: 3 large CUTLASS pairs → same output block → 3-way reduce_noatomic
    run_pipeline_test<float>("Large 3-pair reduce (ij,jk->ik)", "ij,jk->ik",
        CROSS_3PAIR_REDUCE_A, CROSS_3PAIR_REDUCE_B, 1e-3);

    // Test 3: Direct-write + multi-pair — singleton vs multi-pair in same run
    run_pipeline_test<float>("Direct-write + multi-pair (ij,jk->ik)", "ij,jk->ik",
        CROSS_DIRECT_MULTI_A, CROSS_DIRECT_MULTI_B, 1e-4);

    // Test 4: Mixed output block — volume + CUTLASS coexist (no promotion)
    run_pipeline_test<float>("Mixed volume+CUTLASS (ij,jk->ik)", "ij,jk->ik",
        CROSS_PROMOTION_A, CROSS_PROMOTION_B, 1e-3);
    run_timing_stats_test<float>("Stats: mixed volume+CUTLASS (large=true, medium=true)",
        "ij,jk->ik", CROSS_PROMOTION_A, CROSS_PROMOTION_B,
        /*large=*/true, /*medium=*/true, /*small=*/false, /*transpose=*/false);

    // Test 5: Batch=4 CUTLASS
    run_pipeline_test<float>("Batch=4 CUTLASS (bij,bjk->bik)", "bij,bjk->bik",
        CROSS_BATCH4_A, CROSS_BATCH4_B, 1e-4);

    // Test 6: Batch=4 mixed tiers — large + small simultaneously
    run_pipeline_test<float>("Batch=4 mixed tiers (bij,bjk->bik)", "bij,bjk->bik",
        CROSS_BATCH4_MIXED_A, CROSS_BATCH4_MIXED_B, 1e-4);

    // Test 7: Three-tier multi-pair — accumulation in all tiers simultaneously
    run_pipeline_test<float>("Three-tier multi-pair (ij,jk->ik)", "ij,jk->ik",
        CROSS_THREE_TIER_MULTI_A, CROSS_THREE_TIER_MULTI_B, 1e-4);

    // Test 8: Batch=4 GenericPermute — batched GP transpose + CUTLASS
    run_pipeline_test<float>("Batch=4 GP (bjik,bjkl->bil)", "bjik,bjkl->bil",
        CROSS_BATCH4_GP_A, CROSS_BATCH4_GP_B, 1e-4);

    // Test 9: GP mixed trivial across tiers — large + small with GP trivial detection
    run_pipeline_test<float>("GP mixed trivial + tiers (jik,jkl->il)", "jik,jkl->il",
        CROSS_GP_TIERS_A, CROSS_GP_TIERS_B, 1e-3);

    // --- Streaming transpose tests ---
    // Re-run select fuzz seeds with small transpose_buf_cap to force multi-batch streaming.
    // Streaming only activates when: large work exists AND no small/medium needs transpose.
    // We use a very low mn_threshold (1) to route everything to large tier.
    std::cout << "\n--- Streaming Transpose Tests ---\n";

    auto run_streaming_test = [&](const std::string& name, uint64_t seed,
                                  int64_t buf_cap, int64_t mn_threshold_val = 1) {
        std::cout << "Test: " << name << " ... " << std::flush;
        try {
            std::mt19937 meta_rng(seed);
            auto params = randomize_params(meta_rng, 5, 5, 64, 0.3, seed);
            DataType dtype = DataType::FLOAT32;
            auto rc = generate_random_contraction(params, dtype);
            auto expr = parse_einsum(rc.einsum_str);
            std::vector<std::vector<int64_t>> shapes = {
                block_utils::compute_full_shape(rc.A),
                block_utils::compute_full_shape(rc.B)
            };
            validate_einsum(expr, shapes);
            EinsumIndexSets index_sets;
            create_einsum_index_sets(expr, index_sets);
            auto A_gpu = to_gpu_input<float>(rc.A);
            auto B_gpu = to_gpu_input<float>(rc.B);

            auto A_dev = copy_sparse_to_device(A_gpu);
            auto B_dev = copy_sparse_to_device(B_gpu);

            // Reference: no streaming
            DispatchConfig dc_ref;
            dc_ref.mn_threshold = mn_threshold_val;
            auto ref = execute_general_sparse_pipeline<float>(
                A_gpu, B_gpu, A_dev, B_dev, index_sets, expr.output_labels, dc_ref);

            // Streaming
            DispatchConfig dc_stream;
            dc_stream.mn_threshold = mn_threshold_val;
            dc_stream.transpose_buf_cap = buf_cap;
            auto streamed = execute_general_sparse_pipeline<float>(
                A_gpu, B_gpu, A_dev, B_dev, index_sets, expr.output_labels, dc_stream);

            if (ref.total_output_elements != streamed.total_output_elements) {
                std::cout << "FAILED (output size mismatch: " << ref.total_output_elements
                          << " vs " << streamed.total_output_elements << ")\n";
                g_failed++;
                return;
            }
            if (ref.total_output_elements == 0) {
                std::cout << "PASSED (empty)\n";
                g_passed++;
                return;
            }

            // Compare outputs
            std::vector<float> h_ref(ref.total_output_elements), h_str(ref.total_output_elements);
            cudaMemcpy(h_ref.data(), ref.d_output, ref.total_output_elements * sizeof(float),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_str.data(), streamed.d_output, streamed.total_output_elements * sizeof(float),
                       cudaMemcpyDeviceToHost);

            float max_diff = 0;
            for (int64_t i = 0; i < ref.total_output_elements; ++i) {
                float diff = std::abs(h_ref[i] - h_str[i]);
                max_diff = std::max(max_diff, diff);
            }
            if (max_diff == 0.0f) {
                std::cout << "PASSED (bit-identical, einsum=" << rc.einsum_str << ")\n";
                g_passed++;
            } else {
                std::cout << "FAILED (max_diff=" << max_diff << " einsum=" << rc.einsum_str << ")\n";
                g_failed++;
            }
        } catch (const std::exception& e) {
            std::cout << "FAILED (exception: " << e.what() << ")\n";
            g_failed++;
        }
    };

    // Test seeds that produce large-only work with transpose
    run_streaming_test("Streaming: seed 42, buf_cap=1024", 42, 1024, 1);
    run_streaming_test("Streaming: seed 42, buf_cap=4096", 42, 4096);
    run_streaming_test("Streaming: seed 77, buf_cap=2048", 77, 2048);
    run_streaming_test("Streaming: seed 999, buf_cap=512", 999, 512);
    // Test with buf_cap=0 (disabled, should be identical to reference)
    run_streaming_test("Streaming: seed 42, buf_cap=0 (disabled)", 42, 0);

    // Test 10: Streaming GP deterministic — GP transpose with/without streaming must match
    {
        auto run_streaming_gp_test = [&](const std::string& name,
                                          const std::string& einsum_str,
                                          const std::string& tensor_A_content,
                                          const std::string& tensor_B_content,
                                          int64_t buf_cap) {
            std::cout << "Test: " << name << " ... " << std::flush;
            try {
                std::string fileA = create_temp_tensor(tensor_A_content);
                std::string fileB = create_temp_tensor(tensor_B_content);
                auto tensors = parse_tensor_files({fileA, fileB}, DataType::FLOAT32);
                auto expr = parse_einsum(einsum_str);
                std::vector<std::vector<int64_t>> shapes = {
                    block_utils::compute_full_shape(tensors[0]),
                    block_utils::compute_full_shape(tensors[1])
                };
                validate_einsum(expr, shapes);
                EinsumIndexSets index_sets;
                create_einsum_index_sets(expr, index_sets);
                auto A_gpu = to_gpu_input<float>(tensors[0]);
                auto B_gpu = to_gpu_input<float>(tensors[1]);
                std::remove(fileA.c_str());
                std::remove(fileB.c_str());

                auto A_dev = copy_sparse_to_device(A_gpu);
                auto B_dev = copy_sparse_to_device(B_gpu);

                // Reference: no streaming
                DispatchConfig dc_ref;
                auto ref = execute_general_sparse_pipeline<float>(
                    A_gpu, B_gpu, A_dev, B_dev, index_sets, expr.output_labels, dc_ref);

                // Streaming
                DispatchConfig dc_stream;
                dc_stream.transpose_buf_cap = buf_cap;
                auto streamed = execute_general_sparse_pipeline<float>(
                    A_gpu, B_gpu, A_dev, B_dev, index_sets, expr.output_labels, dc_stream);

                if (ref.total_output_elements != streamed.total_output_elements) {
                    std::cout << "FAILED (size mismatch)\n";
                    g_failed++;
                    return;
                }
                if (ref.total_output_elements == 0) {
                    std::cout << "PASSED (empty)\n";
                    g_passed++;
                    return;
                }

                std::vector<float> h_ref(ref.total_output_elements);
                std::vector<float> h_str(ref.total_output_elements);
                cudaMemcpy(h_ref.data(), ref.d_output,
                           ref.total_output_elements * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_str.data(), streamed.d_output,
                           streamed.total_output_elements * sizeof(float), cudaMemcpyDeviceToHost);

                float max_diff = 0;
                for (int64_t i = 0; i < ref.total_output_elements; ++i)
                    max_diff = std::max(max_diff, std::abs(h_ref[i] - h_str[i]));

                if (max_diff == 0.0f) {
                    std::cout << "PASSED (bit-identical)\n";
                    g_passed++;
                } else {
                    std::cout << "FAILED (max_diff=" << max_diff << ")\n";
                    g_failed++;
                }
            } catch (const std::exception& e) {
                std::cout << "FAILED (exception: " << e.what() << ")\n";
                g_failed++;
            }
        };

        run_streaming_gp_test("Streaming GP: mixed trivial (buf_cap=4096)",
            "jik,jkl->il", CROSS_GP_MIXED_TRIVIAL_A, CROSS_GP_MIXED_TRIVIAL_B, 4096);
    }

    // ========================================================
    // Many-shapes tests: verify uint32 shape_idx with >65536 unique pair shapes
    // ========================================================
    // Synthetic "ab,bc->ac" contractions with all values=1.0.
    // Every output element = sum of K section sizes (analytically verifiable).
    // Three tier targets: volume (medium), warp (small), large (CUTLASS+hypertile).
    if (!g_f64) {
        auto run_many_shapes_test = [](const char* name,
                                       const std::vector<int>& a_sizes,  // M per section
                                       const std::vector<int>& b_sizes,  // K per section
                                       const std::vector<int>& c_sizes)  // N per section
        {
            std::cout << "Test: " << name << " ... " << std::flush;
            try {
                const int n_a = static_cast<int>(a_sizes.size());
                const int n_b = static_cast<int>(b_sizes.size());
                const int n_c = static_cast<int>(c_sizes.size());

                // Expected output value: every element = sum of K section sizes
                float expected_val = 0;
                for (int k : b_sizes) expected_val += static_cast<float>(k);

                // Build GPUSparseInput for A (rank 2: a,b) — all blocks present
                GPUSparseInput<float> A_gpu;
                A_gpu.rank = 2;
                A_gpu.num_blocks = n_a * n_b;
                A_gpu.labels = {'a', 'b'};
                A_gpu.num_blocks_per_dim = {n_a, n_b};

                int64_t full_a = 0;
                for (int s : a_sizes) full_a += s;
                int64_t full_b = 0;
                for (int s : b_sizes) full_b += s;
                int64_t full_c = 0;
                for (int s : c_sizes) full_c += s;
                A_gpu.full_extents = {full_a, full_b};

                // Block sizes per dim
                A_gpu.block_sizes_offsets = {0, n_a};
                A_gpu.block_sizes_flat.reserve(n_a + n_b);
                for (int s : a_sizes) A_gpu.block_sizes_flat.push_back(s);
                for (int s : b_sizes) A_gpu.block_sizes_flat.push_back(s);

                // Blocks: enumerate all (i,j) combos, row-major order
                A_gpu.block_coords.reserve(A_gpu.num_blocks * 2);
                A_gpu.block_dims.reserve(A_gpu.num_blocks * 2);
                A_gpu.block_offsets.reserve(A_gpu.num_blocks);
                A_gpu.block_volumes.reserve(A_gpu.num_blocks);
                int64_t a_offset = 0;
                for (int i = 0; i < n_a; ++i) {
                    for (int j = 0; j < n_b; ++j) {
                        A_gpu.block_coords.push_back(i);
                        A_gpu.block_coords.push_back(j);
                        A_gpu.block_dims.push_back(a_sizes[i]);
                        A_gpu.block_dims.push_back(b_sizes[j]);
                        A_gpu.block_offsets.push_back(a_offset);
                        int64_t vol = static_cast<int64_t>(a_sizes[i]) * b_sizes[j];
                        A_gpu.block_volumes.push_back(vol);
                        a_offset += vol;
                    }
                }
                A_gpu.total_values = a_offset;

                // Build GPUSparseInput for B (rank 2: b,c)
                GPUSparseInput<float> B_gpu;
                B_gpu.rank = 2;
                B_gpu.num_blocks = n_b * n_c;
                B_gpu.labels = {'b', 'c'};
                B_gpu.num_blocks_per_dim = {n_b, n_c};
                B_gpu.full_extents = {full_b, full_c};
                B_gpu.block_sizes_offsets = {0, n_b};
                B_gpu.block_sizes_flat.reserve(n_b + n_c);
                for (int s : b_sizes) B_gpu.block_sizes_flat.push_back(s);
                for (int s : c_sizes) B_gpu.block_sizes_flat.push_back(s);

                B_gpu.block_coords.reserve(B_gpu.num_blocks * 2);
                B_gpu.block_dims.reserve(B_gpu.num_blocks * 2);
                B_gpu.block_offsets.reserve(B_gpu.num_blocks);
                B_gpu.block_volumes.reserve(B_gpu.num_blocks);
                int64_t b_offset = 0;
                for (int j = 0; j < n_b; ++j) {
                    for (int k = 0; k < n_c; ++k) {
                        B_gpu.block_coords.push_back(j);
                        B_gpu.block_coords.push_back(k);
                        B_gpu.block_dims.push_back(b_sizes[j]);
                        B_gpu.block_dims.push_back(c_sizes[k]);
                        B_gpu.block_offsets.push_back(b_offset);
                        int64_t vol = static_cast<int64_t>(b_sizes[j]) * c_sizes[k];
                        B_gpu.block_volumes.push_back(vol);
                        b_offset += vol;
                    }
                }
                B_gpu.total_values = b_offset;

                // Allocate host values (all 1.0) via values vector
                A_gpu.values.assign(a_offset, 1.0f);
                B_gpu.values.assign(b_offset, 1.0f);

                // Upload to device
                auto A_dev = copy_sparse_to_device(A_gpu);
                auto B_dev = copy_sparse_to_device(B_gpu);

                // Build einsum config
                EinsumIndexSets index_sets;
                index_sets.free_indices_A = {'a'};
                index_sets.free_indices_B = {'c'};
                index_sets.contracted_indices = {'b'};
                std::string output_labels = "ac";

                // Run pipeline
                DispatchConfig dc;
                PipelineTimingBreakdown timing{};
                auto result = execute_general_sparse_pipeline<float>(
                    A_gpu, B_gpu, A_dev, B_dev, index_sets, output_labels, dc, &timing);

                if (result.total_output_elements == 0) {
                    std::cout << "FAILED (empty output)\n";
                    g_failed++;
                    return;
                }

                // Verify: every output element should equal expected_val
                std::vector<float> h_out(result.total_output_elements);
                cudaMemcpy(h_out.data(), result.d_output,
                           result.total_output_elements * sizeof(float),
                           cudaMemcpyDeviceToHost);

                float max_err = 0;
                int bad_count = 0;
                for (int64_t i = 0; i < result.total_output_elements; ++i) {
                    float err = std::abs(h_out[i] - expected_val);
                    float rtol = std::abs(expected_val) * 1e-4f + 1e-5f;
                    if (err > rtol) {
                        if (bad_count < 3 && g_verbose) {
                            printf("  elem %ld: got %.6f expected %.6f (err=%.6e)\n",
                                   (long)i, h_out[i], expected_val, err);
                        }
                        bad_count++;
                    }
                    max_err = std::max(max_err, err);
                }

                int total_pairs = n_a * n_b * n_c;
                int unique_shapes = total_pairs;  // each pair has unique (M,K,N)
                if (bad_count == 0) {
                    printf("PASSED (%d pairs, %d shapes, L=%d M=%d S=%d, %.1fms)\n",
                           total_pairs, unique_shapes,
                           timing.large_gemm_problems, timing.medium_descs, timing.small_descs,
                           timing.total_ms);
                    g_passed++;
                } else {
                    printf("FAILED (%d/%ld bad, max_err=%.3e, L=%d M=%d S=%d)\n",
                           bad_count, (long)result.total_output_elements, max_err,
                           timing.large_gemm_problems, timing.medium_descs, timing.small_descs);
                    g_failed++;
                }
            } catch (const std::exception& e) {
                std::cout << "FAILED (exception: " << e.what() << ")\n";
                g_failed++;
            }
        };

        // --- Volume tier: M,N ∈ [4,32], K ∈ [1,80] → all medium ---
        // 29 * 80 * 29 = 67,280 unique pair shapes > 65536
        {
            std::vector<int> a_sizes(29), b_sizes(80), c_sizes(29);
            for (int i = 0; i < 29; ++i) a_sizes[i] = 4 + i;   // 4..32
            for (int i = 0; i < 80; ++i) b_sizes[i] = 1 + i;   // 1..80
            for (int i = 0; i < 29; ++i) c_sizes[i] = 4 + i;   // 4..32
            run_many_shapes_test("Many shapes: volume (67K unique)", a_sizes, b_sizes, c_sizes);
        }

        // --- Warp tier: M ∈ {1,2,3}, K ∈ [1,7300], N ∈ {1,2,3} → all small ---
        // 3 * 7300 * 3 = 65,700 unique pair shapes > 65536
        {
            std::vector<int> a_sizes = {1, 2, 3};
            std::vector<int> b_sizes(7300);
            for (int i = 0; i < 7300; ++i) b_sizes[i] = 1 + i;  // 1..7300
            std::vector<int> c_sizes = {1, 2, 3};
            run_many_shapes_test("Many shapes: warp (66K unique)", a_sizes, b_sizes, c_sizes);
        }

        // --- Large tier: M,N ∈ [33,52], K = 1 → all CUTLASS ---
        // 20 * 1 * 20 = 400 unique shapes (exercises uint32 BlockShapeMap path,
        // though not >65536 — dense output size limits this tier).
        {
            std::vector<int> a_sizes(20), c_sizes(20);
            for (int i = 0; i < 20; ++i) a_sizes[i] = 33 + i;  // 33..52
            std::vector<int> b_sizes = {1};
            for (int i = 0; i < 20; ++i) c_sizes[i] = 33 + i;  // 33..52
            run_many_shapes_test("Many shapes: large (400 shapes, M>32)", a_sizes, b_sizes, c_sizes);
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "Results: " << g_passed << " passed, " << g_failed << " failed\n";
    std::cout << "========================================\n";

    return g_failed > 0 ? 1 : 0;
}
