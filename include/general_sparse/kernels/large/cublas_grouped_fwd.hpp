#pragma once

// Forward declarations for the cuBLAS grouped-GEMM Large-tier backend.
// Includes no cuBLAS headers, so pipeline.cuh stays cheap to rebuild; the
// implementation lives in src/cublas/cublas_grouped_gemm.cu.
//
// The backend reuses CutlassGemmRunnerResult verbatim: same pooled device/pinned
// arg buffers, same run(stream) closure, same A/B pointer-patch contract used by
// the post-transpose staging path.  Only the buffer's internal region layout and
// the launch differ.

#include "cutlass_grouped_fwd.hpp"

namespace bsptc {

// Byte size of the cuBLAS arg-region layout for `count` expanded problems.
// Declared here so allocate_cutlass_args() can size a shared buffer that fits
// whichever backend the per-queue selector picks (the streaming arg buffers are
// allocated once and reused across queues that may differ in backend).
//
// Device side: Aarray / Barray / Carray.
// Host side:   the same three pointer arrays (H2D staging) plus the per-group
//              scalar arrays cuBLAS reads directly from host memory.
size_t cublas_args_device_bytes(int count);
size_t cublas_args_host_bytes(int count);

// True when a cuBLAS grouped-GEMM candidate may be considered at all: the
// runtime exposes cublas{S,D}gemmGroupedBatched and a handle can be created.
bool is_cublas_grouped_gemm_available();

template <typename T>
CutlassGemmRunnerResult prepare_cublas_gemm(
    const LargeBlockWork<T>& work, T alpha, T beta, cudaStream_t stream);

template <typename T>
void fill_cublas_args(
    CutlassGemmRunnerResult& result, const LargeBlockWork<T>& work,
    T alpha, T beta, cudaStream_t stream);

} // namespace bsptc
