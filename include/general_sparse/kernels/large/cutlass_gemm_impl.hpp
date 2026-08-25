#pragma once

// Shared CUTLASS GEMM implementation templates.
// Included by each per-shape .cu file. Contains prepare_cutlass_gemm_impl
// and fill_cutlass_args_impl which are parameterized on the Gemm kernel type.
//
// IMPORTANT: CUTLASS headers must already be included before this file.

#include "cutlass_grouped_fwd.hpp"

#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <functional>

namespace bsptc {

// Generate batch-expanded leading dims from the descs into the pinned staging
// regions (h_ptrs[5..8]). One sequential pass per region: interleaving the
// four streams with the coord loop measurably hurts pinned-memory write
// locality at high problem counts.
template <typename Element>
static void fill_leading_dims(const LargeBlockWork<Element>& work,
                              void* const (&h_ptrs)[9]) {
    auto fill = [&](void* dst, auto pick) {
        auto* out = static_cast<int64_t*>(dst);
        size_t idx = 0;
        for (const auto& prob : work.problems)
            for (int64_t b = 0; b < prob.batch_size; ++b)
                out[idx++] = pick(prob);
    };
    fill(h_ptrs[5], [](const GemmProblemDesc& p) { return p.lda; });
    fill(h_ptrs[6], [](const GemmProblemDesc& p) { return p.ldb; });
    fill(h_ptrs[7], [](const GemmProblemDesc& p) { return p.ldc; });
    fill(h_ptrs[8], [](const GemmProblemDesc& p) { return p.ldc; });  // D = C
}

// Prepare CUTLASS GEMM: allocate unified arg buffer + pinned staging, fill,
// async H2D, construct Gemm::Arguments.  Returns runner + buffers to free.
// No workspace (kDeviceOnly scheduling).
template <typename Gemm, typename Element>
static CutlassGemmRunnerResult prepare_cutlass_gemm_impl(
    const LargeBlockWork<Element>& work,
    Element alpha,
    Element beta,
    cudaStream_t stream)
{
    CutlassGemmRunnerResult result;

    const int problem_count = static_cast<int>(work.num_gemm_problems);
    if (problem_count <= 0) return result;

    const size_t sizes[9] = {
        static_cast<size_t>(problem_count) * sizeof(cutlass::gemm::GemmCoord),
        static_cast<size_t>(problem_count) * sizeof(Element*),
        static_cast<size_t>(problem_count) * sizeof(Element*),
        static_cast<size_t>(problem_count) * sizeof(Element*),
        static_cast<size_t>(problem_count) * sizeof(Element*),
        static_cast<size_t>(problem_count) * sizeof(int64_t),
        static_cast<size_t>(problem_count) * sizeof(int64_t),
        static_cast<size_t>(problem_count) * sizeof(int64_t),
        static_cast<size_t>(problem_count) * sizeof(int64_t),
    };
    const size_t d_total = total_allocation_size(sizes);
    const size_t h_total = d_total + sizes[0]; // staging + persistent h_problem_sizes

    // Acquire device and pinned host buffers from the process-wide pools
    // (avoids ~1 ms cudaMallocHost page-pinning per call)
    result.d_args_buf = device_args_pool().acquire(d_total);
    result.h_args_buf = pinned_args_pool().acquire(h_total);

    // Resolve device pointers
    void* d_ptrs[9];
    resolve_pointers(result.d_args_buf, sizes, d_ptrs);

    // Resolve host staging pointers (same layout)
    void* h_ptrs[9];
    resolve_pointers(result.h_args_buf, sizes, h_ptrs);

    // Persistent host problem_sizes lives after the staging region
    auto* h_problem_sizes = reinterpret_cast<cutlass::gemm::GemmCoord*>(
        static_cast<char*>(result.h_args_buf) + d_total);

    // Fill staging buffer
    auto* h_staging_ps = static_cast<cutlass::gemm::GemmCoord*>(h_ptrs[0]);
    size_t prob_idx = 0;
    for (const auto& prob : work.problems) {
        for (int64_t b = 0; b < prob.batch_size; ++b) {
            auto coord = cutlass::gemm::GemmCoord(
                static_cast<int>(prob.M),
                static_cast<int>(prob.N),
                static_cast<int>(prob.K));
            h_staging_ps[prob_idx] = coord;
            h_problem_sizes[prob_idx] = coord;
            ++prob_idx;
        }
    }
    fill_leading_dims(work, h_ptrs);

    // Store patching locations
    result.d_ptr_A_loc = d_ptrs[1];
    result.d_ptr_B_loc = d_ptrs[2];
    result.h_ptr_A_loc = h_ptrs[1];
    result.h_ptr_B_loc = h_ptrs[2];
    result.ptr_array_bytes = sizes[1];  // sizes[1] == sizes[2]

    std::memcpy(h_ptrs[1], work.ptr_A.data(), sizes[1]);
    std::memcpy(h_ptrs[2], work.ptr_B.data(), sizes[2]);
    std::memcpy(h_ptrs[3], work.ptr_C.data(), sizes[3]);
    std::memcpy(h_ptrs[4], work.ptr_C.data(), sizes[4]); // D = C

    // Single async H2D transfer
    check_cuda(cudaMemcpyAsync(result.d_args_buf, result.h_args_buf, d_total,
                                cudaMemcpyHostToDevice, stream),
               "prepare_gemm: memcpy unified");

    // Construct Gemm::Arguments
    int threadblock_count = Gemm::sufficient(h_problem_sizes, problem_count);
    if (threadblock_count == 0) {
        // Drain the in-flight H2D copy before returning buffers to the pool
        // (pool release does not synchronize, unlike cudaFreeHost).
        cudaStreamSynchronize(stream);
        result.free_buffers();
        throw std::runtime_error("CUTLASS: insufficient resources for grouped GEMM");
    }

    typename Gemm::EpilogueOutputOp::Params epilogue_params(alpha, beta);

    typename Gemm::Arguments args(
        static_cast<cutlass::gemm::GemmCoord*>(d_ptrs[0]),
        problem_count,
        threadblock_count,
        epilogue_params,
        static_cast<Element**>(d_ptrs[1]),
        static_cast<Element**>(d_ptrs[2]),
        static_cast<Element**>(d_ptrs[3]),
        static_cast<Element**>(d_ptrs[4]),
        static_cast<int64_t*>(d_ptrs[5]),
        static_cast<int64_t*>(d_ptrs[6]),
        static_cast<int64_t*>(d_ptrs[7]),
        static_cast<int64_t*>(d_ptrs[8]),
        h_problem_sizes
    );

    // Runner: just initialize + run, no allocations (kDeviceOnly: workspace = nullptr)
    result.run = [args](cudaStream_t s) mutable {
        Gemm gemm_op;
        cutlass::Status status = gemm_op.initialize(args, nullptr, s);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("CUTLASS grouped GEMM initialization failed");
        }
        // Clear any pre-existing CUDA error before launch
        (void)cudaGetLastError();
        status = gemm_op.run(s);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("CUTLASS grouped GEMM execution failed");
        }
    };

    return result;
}

// Fill pre-allocated arg buffers (for double-buffered streaming).
template <typename Gemm, typename Element>
static void fill_cutlass_args_impl(
    CutlassGemmRunnerResult& result,
    const LargeBlockWork<Element>& work,
    Element alpha,
    Element beta,
    cudaStream_t stream)
{
    const int problem_count = static_cast<int>(work.num_gemm_problems);
    if (problem_count <= 0) { result.run = nullptr; return; }

    const size_t sizes[9] = {
        static_cast<size_t>(problem_count) * sizeof(cutlass::gemm::GemmCoord),
        static_cast<size_t>(problem_count) * sizeof(Element*),
        static_cast<size_t>(problem_count) * sizeof(Element*),
        static_cast<size_t>(problem_count) * sizeof(Element*),
        static_cast<size_t>(problem_count) * sizeof(Element*),
        static_cast<size_t>(problem_count) * sizeof(int64_t),
        static_cast<size_t>(problem_count) * sizeof(int64_t),
        static_cast<size_t>(problem_count) * sizeof(int64_t),
        static_cast<size_t>(problem_count) * sizeof(int64_t),
    };
    const size_t d_total = total_allocation_size(sizes);

    void* d_ptrs[9];
    resolve_pointers(result.d_args_buf, sizes, d_ptrs);
    void* h_ptrs[9];
    resolve_pointers(result.h_args_buf, sizes, h_ptrs);

    auto* h_problem_sizes = reinterpret_cast<cutlass::gemm::GemmCoord*>(
        static_cast<char*>(result.h_args_buf) + d_total);

    auto* h_staging_ps = static_cast<cutlass::gemm::GemmCoord*>(h_ptrs[0]);
    size_t prob_idx = 0;
    for (const auto& prob : work.problems) {
        for (int64_t b = 0; b < prob.batch_size; ++b) {
            auto coord = cutlass::gemm::GemmCoord(
                static_cast<int>(prob.M), static_cast<int>(prob.N), static_cast<int>(prob.K));
            h_staging_ps[prob_idx] = coord;
            h_problem_sizes[prob_idx] = coord;
            ++prob_idx;
        }
    }
    fill_leading_dims(work, h_ptrs);

    result.d_ptr_A_loc = d_ptrs[1];
    result.d_ptr_B_loc = d_ptrs[2];
    result.h_ptr_A_loc = h_ptrs[1];
    result.h_ptr_B_loc = h_ptrs[2];
    result.ptr_array_bytes = sizes[1];

    std::memcpy(h_ptrs[1], work.ptr_A.data(), sizes[1]);
    std::memcpy(h_ptrs[2], work.ptr_B.data(), sizes[2]);
    std::memcpy(h_ptrs[3], work.ptr_C.data(), sizes[3]);
    std::memcpy(h_ptrs[4], work.ptr_C.data(), sizes[4]);

    check_cuda(cudaMemcpyAsync(result.d_args_buf, result.h_args_buf, d_total,
                                cudaMemcpyHostToDevice, stream),
               "fill_args: memcpy");

    int threadblock_count = Gemm::sufficient(h_problem_sizes, problem_count);
    if (threadblock_count == 0) {
        throw std::runtime_error("CUTLASS: insufficient resources for grouped GEMM");
    }

    typename Gemm::EpilogueOutputOp::Params epilogue_params(alpha, beta);
    typename Gemm::Arguments args(
        static_cast<cutlass::gemm::GemmCoord*>(d_ptrs[0]),
        problem_count, threadblock_count, epilogue_params,
        static_cast<Element**>(d_ptrs[1]), static_cast<Element**>(d_ptrs[2]),
        static_cast<Element**>(d_ptrs[3]), static_cast<Element**>(d_ptrs[4]),
        static_cast<int64_t*>(d_ptrs[5]), static_cast<int64_t*>(d_ptrs[6]),
        static_cast<int64_t*>(d_ptrs[7]), static_cast<int64_t*>(d_ptrs[8]),
        h_problem_sizes
    );

    result.run = [args](cudaStream_t s) mutable {
        Gemm gemm_op;
        cutlass::Status status = gemm_op.initialize(args, nullptr, s);
        if (status != cutlass::Status::kSuccess)
            throw std::runtime_error("CUTLASS grouped GEMM initialization failed");
        (void)cudaGetLastError();
        status = gemm_op.run(s);
        if (status != cutlass::Status::kSuccess)
            throw std::runtime_error("CUTLASS grouped GEMM execution failed");
    };
}

} // namespace bsptc
