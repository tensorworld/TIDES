// Benchmark: cuTENSOR permutation vs hypertile transpose.
//
// Two comparison modes:
//   1. Single large tensor: one dense block, sweep volume and rank
//   2. Batched small tensors: many blocks of the same shape, sweep block count
//
// cuTENSOR permutation: cutensorCreatePermutation → cutensorCreatePlan → cutensorPermute
// Hypertile:            build_task_for_shape → pack_shape → launch_hypertile_kernel
//
// Usage:
//   ./bench_transpose_cutensor [--warmup N] [--repeat N] [--dtype f32|f64]
//   ./bench_transpose_cutensor --batch-sweep [--total-vol N] [--dtype f32|f64]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>

#include "bs_parser.hpp"
#include "einsum.hpp"
#include "general_sparse/gemm_layout_check.hpp"
#include "general_sparse/gpu_sparse_device.cuh"
#include "general_sparse/kernels/transpose/nd_hypertile.cuh"

using namespace bsptc;

// ---------------------------------------------------------------------------
// cuTENSOR helpers
// ---------------------------------------------------------------------------

static void check_cutensor(cutensorStatus_t status, const char* msg) {
    if (status != CUTENSOR_STATUS_SUCCESS) {
        fprintf(stderr, "cuTENSOR error: %s: %s\n", msg, cutensorGetErrorString(status));
        exit(1);
    }
}

static cutensorHandle_t get_handle() {
    static cutensorHandle_t h = []() {
        cutensorHandle_t handle{};
        check_cutensor(cutensorCreate(&handle), "cutensorCreate");
        return handle;
    }();
    return h;
}

// ---------------------------------------------------------------------------
// Timing harness
// ---------------------------------------------------------------------------

struct BenchResult { float median_ms, min_ms, max_ms; };

template <typename Fn>
BenchResult gpu_bench(Fn fn, int warmup, int repeat, cudaStream_t stream) {
    for (int i = 0; i < warmup; i++) fn();
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> times(repeat);
    for (int i = 0; i < repeat; i++) {
        cudaEventRecord(start, stream);
        fn();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::sort(times.begin(), times.end());
    return {times[repeat / 2], times[0], times[repeat - 1]};
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Factor volume into roughly-equal dims.
static std::vector<int64_t> factor_dims(int64_t volume, int rank) {
    std::vector<int64_t> dims(rank);
    double base = std::pow(static_cast<double>(volume), 1.0 / rank);
    int64_t prod = 1;
    for (int i = 0; i < rank - 1; i++) {
        dims[i] = std::max(int64_t(2), static_cast<int64_t>(std::round(base)));
        prod *= dims[i];
    }
    dims[rank - 1] = std::max(int64_t(1), volume / prod);
    return dims;
}

// Generate irregular dims with a deterministic seed.
// Produces shapes like (3, 128, 7) — wide aspect ratios, prime/odd sizes.
static std::vector<int64_t> irregular_dims(int64_t approx_volume, int rank, uint32_t seed) {
    std::mt19937 rng(seed * 999983u + static_cast<uint32_t>(rank) * 7u +
                     static_cast<uint32_t>(approx_volume & 0xFFFF));
    // Pool of irregular dimension sizes: primes, odd, non-power-of-2
    static const int64_t dim_pool[] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
        47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
        127, 128, 255, 256, 511, 512, 1023, 1024
    };
    constexpr int pool_size = sizeof(dim_pool) / sizeof(dim_pool[0]);

    // Pick random dims, then scale the last dim to hit approximate volume.
    std::vector<int64_t> dims(rank);
    int64_t prod = 1;
    for (int i = 0; i < rank - 1; i++) {
        // Bias smaller dims for higher ranks to keep volume in range
        int max_idx = (rank <= 3) ? pool_size - 1 :
                      (rank <= 5) ? pool_size - 5 : pool_size - 9;
        max_idx = std::max(max_idx, 10);
        std::uniform_int_distribution<int> dist(0, max_idx);
        dims[i] = dim_pool[dist(rng)];
        prod *= dims[i];
    }
    // Last dim absorbs remainder to hit target volume
    dims[rank - 1] = std::max(int64_t(2), approx_volume / prod);
    return dims;
}

static std::string dims_to_string(const std::vector<int64_t>& dims) {
    std::string s;
    for (size_t i = 0; i < dims.size(); i++) {
        if (i > 0) s += "x";
        s += std::to_string(dims[i]);
    }
    return s;
}

static int64_t product(const std::vector<int64_t>& v) {
    int64_t p = 1;
    for (auto x : v) p *= x;
    return p;
}

// Generate a random non-identity permutation using a deterministic seed.
// Seed is combined with rank so different ranks get different perms.
static std::vector<int32_t> make_random_perm(int rank, uint32_t seed) {
    std::mt19937 rng(seed * 1000003u + static_cast<uint32_t>(rank));
    std::vector<int32_t> perm(rank);
    std::iota(perm.begin(), perm.end(), 0);
    // Shuffle until non-identity
    for (int attempt = 0; attempt < 100; attempt++) {
        std::shuffle(perm.begin(), perm.end(), rng);
        bool is_identity = true;
        for (int i = 0; i < rank; i++)
            if (perm[i] != i) { is_identity = false; break; }
        if (!is_identity) break;
    }
    return perm;
}

static std::string perm_to_string(const std::vector<int32_t>& perm) {
    std::string s;
    for (size_t i = 0; i < perm.size(); i++) {
        if (i > 0) s += ",";
        s += std::to_string(perm[i]);
    }
    return s;
}

// Compare cuTENSOR and hypertile outputs element-wise.
// Returns number of mismatches (0 = pass).
template <typename T>
static int verify_against_cutensor(const T* d_ct, const T* d_ht, int64_t vol,
                                   const char* shape_str) {
    std::vector<T> h_ct(vol), h_ht(vol);
    cudaMemcpy(h_ct.data(), d_ct, vol * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ht.data(), d_ht, vol * sizeof(T), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (int64_t i = 0; i < vol; i++) {
        if (h_ct[i] != h_ht[i]) {
            if (mismatches < 3) {
                printf("  MISMATCH [%s] at %ld: hypertile=%f, cuTENSOR=%f\n",
                       shape_str, (long)i, (double)h_ht[i], (double)h_ct[i]);
            }
            mismatches++;
        }
    }
    return mismatches;
}

// ---------------------------------------------------------------------------
// cuTENSOR single-tensor permutation benchmark
// ---------------------------------------------------------------------------

template <typename T>
static float bench_cutensor_permute(
    const std::vector<int64_t>& dims,
    const std::vector<int32_t>& perm,
    const T* d_src, T* d_dst,
    int warmup, int repeat, cudaStream_t stream)
{
    cutensorHandle_t handle = get_handle();
    const uint32_t rank = static_cast<uint32_t>(dims.size());

    // Row-major strides
    std::vector<int64_t> src_strides(rank);
    src_strides[rank - 1] = 1;
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d)
        src_strides[d] = src_strides[d + 1] * dims[d + 1];

    // Dest shape and strides
    std::vector<int64_t> dst_dims(rank);
    for (uint32_t d = 0; d < rank; d++) dst_dims[d] = dims[perm[d]];
    std::vector<int64_t> dst_strides(rank);
    dst_strides[rank - 1] = 1;
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d)
        dst_strides[d] = dst_strides[d + 1] * dst_dims[d + 1];

    // Mode labels: source uses 0,1,...,rank-1; dest uses perm[0],perm[1],...
    std::vector<int32_t> modeA(rank), modeB(rank);
    for (uint32_t d = 0; d < rank; d++) modeA[d] = static_cast<int32_t>(d);
    for (uint32_t d = 0; d < rank; d++) modeB[d] = perm[d];

    cudaDataType_t dtype = std::is_same<T, float>::value ? CUDA_R_32F : CUDA_R_64F;
    cutensorComputeDescriptor_t compute = std::is_same<T, float>::value
        ? CUTENSOR_COMPUTE_DESC_32F : CUTENSOR_COMPUTE_DESC_64F;

    // Single-tensor: pointers are always from cudaMalloc (256-byte aligned).
    cutensorTensorDescriptor_t descA{}, descB{};
    check_cutensor(cutensorCreateTensorDescriptor(handle, &descA, rank,
        dims.data(), src_strides.data(), dtype, 256), "descA");
    check_cutensor(cutensorCreateTensorDescriptor(handle, &descB, rank,
        dst_dims.data(), dst_strides.data(), dtype, 256), "descB");

    cutensorOperationDescriptor_t opDesc{};
    check_cutensor(cutensorCreatePermutation(handle, &opDesc,
        descA, modeA.data(), CUTENSOR_OP_IDENTITY,
        descB, modeB.data(), compute), "createPermutation");

    cutensorPlanPreference_t pref{};
    check_cutensor(cutensorCreatePlanPreference(handle, &pref,
        CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE), "createPlanPref");

    uint64_t ws_size = 0;
    check_cutensor(cutensorEstimateWorkspaceSize(handle, opDesc, pref,
        CUTENSOR_WORKSPACE_DEFAULT, &ws_size), "estimateWorkspace");

    cutensorPlan_t plan{};
    check_cutensor(cutensorCreatePlan(handle, &plan, opDesc, pref, ws_size), "createPlan");

    void* d_workspace = nullptr;
    if (ws_size > 0) cudaMalloc(&d_workspace, ws_size);

    T alpha = T(1);

    auto result = gpu_bench([&]() {
        check_cutensor(cutensorPermute(handle, plan, &alpha,
            d_src, d_dst, stream), "permute");
    }, warmup, repeat, stream);

    if (d_workspace) cudaFree(d_workspace);
    cutensorDestroyPlan(plan);
    cutensorDestroyPlanPreference(pref);
    cutensorDestroyOperationDescriptor(opDesc);
    cutensorDestroyTensorDescriptor(descB);
    cutensorDestroyTensorDescriptor(descA);

    return result.median_ms;
}

// ---------------------------------------------------------------------------
// cuTENSOR batched permutation (loop over blocks, single plan)
// ---------------------------------------------------------------------------

template <typename T>
static float bench_cutensor_batched(
    const std::vector<int64_t>& block_dims,
    const std::vector<int32_t>& perm,
    int num_blocks,
    const T* d_src, T* d_dst,
    const std::vector<int64_t>& offsets,
    int warmup, int repeat, cudaStream_t stream)
{
    cutensorHandle_t handle = get_handle();
    const uint32_t rank = static_cast<uint32_t>(block_dims.size());

    // All blocks have the same shape → create one plan, reuse for all blocks.
    std::vector<int64_t> src_strides(rank);
    src_strides[rank - 1] = 1;
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d)
        src_strides[d] = src_strides[d + 1] * block_dims[d + 1];

    std::vector<int64_t> dst_dims(rank);
    for (uint32_t d = 0; d < rank; d++) dst_dims[d] = block_dims[perm[d]];
    std::vector<int64_t> dst_strides(rank);
    dst_strides[rank - 1] = 1;
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d)
        dst_strides[d] = dst_strides[d + 1] * dst_dims[d + 1];

    std::vector<int32_t> modeA(rank), modeB(rank);
    for (uint32_t d = 0; d < rank; d++) modeA[d] = static_cast<int32_t>(d);
    for (uint32_t d = 0; d < rank; d++) modeB[d] = perm[d];

    cudaDataType_t dtype = std::is_same<T, float>::value ? CUDA_R_32F : CUDA_R_64F;
    cutensorComputeDescriptor_t compute = std::is_same<T, float>::value
        ? CUTENSOR_COMPUTE_DESC_32F : CUTENSOR_COMPUTE_DESC_64F;

    // Alignment: block offsets in packed layout may not be 256-byte aligned.
    const uint32_t align = static_cast<uint32_t>(sizeof(T));

    cutensorTensorDescriptor_t descA{}, descB{};
    check_cutensor(cutensorCreateTensorDescriptor(handle, &descA, rank,
        block_dims.data(), src_strides.data(), dtype, align), "descA");
    check_cutensor(cutensorCreateTensorDescriptor(handle, &descB, rank,
        dst_dims.data(), dst_strides.data(), dtype, align), "descB");

    cutensorOperationDescriptor_t opDesc{};
    check_cutensor(cutensorCreatePermutation(handle, &opDesc,
        descA, modeA.data(), CUTENSOR_OP_IDENTITY,
        descB, modeB.data(), compute), "createPermutation");

    cutensorPlanPreference_t pref{};
    check_cutensor(cutensorCreatePlanPreference(handle, &pref,
        CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE), "createPlanPref");

    uint64_t ws_size = 0;
    check_cutensor(cutensorEstimateWorkspaceSize(handle, opDesc, pref,
        CUTENSOR_WORKSPACE_DEFAULT, &ws_size), "estimateWorkspace");

    cutensorPlan_t plan{};
    check_cutensor(cutensorCreatePlan(handle, &plan, opDesc, pref, ws_size), "createPlan");

    void* d_workspace = nullptr;
    if (ws_size > 0) cudaMalloc(&d_workspace, ws_size);

    T alpha = T(1);

    // For very large block counts, cuTENSOR's per-block loop is too slow.
    // Benchmark a subset and extrapolate to avoid extremely long runs.
    int bench_blocks = std::min(num_blocks, 1000);
    float loop_scale = static_cast<float>(num_blocks) / static_cast<float>(bench_blocks);

    auto result = gpu_bench([&]() {
        for (int b = 0; b < bench_blocks; b++) {
            const T* src_ptr = d_src + offsets[b];
            T* dst_ptr = d_dst + offsets[b];
            cutensorPermute(handle, plan, &alpha, src_ptr, dst_ptr, stream);
        }
    }, warmup, repeat, stream);
    result.median_ms *= loop_scale;
    result.min_ms *= loop_scale;
    result.max_ms *= loop_scale;

    if (d_workspace) cudaFree(d_workspace);
    cutensorDestroyPlan(plan);
    cutensorDestroyPlanPreference(pref);
    cutensorDestroyOperationDescriptor(opDesc);
    cutensorDestroyTensorDescriptor(descB);
    cutensorDestroyTensorDescriptor(descA);

    return result.median_ms;
}

// ---------------------------------------------------------------------------
// Hypertile single-block benchmark
// ---------------------------------------------------------------------------

template <typename T>
static float bench_hypertile_single(
    const std::vector<int64_t>& dims,
    const std::vector<int32_t>& perm,  // dst←src (cuTENSOR convention)
    const T* d_src, T* d_dst,
    int warmup, int repeat, cudaStream_t stream)
{
    const int rank = static_cast<int>(dims.size());
    dim_t d[16];
    for (int i = 0; i < rank; i++) d[i] = static_cast<dim_t>(dims[i]);

    // perm is dst←src (cuTENSOR modeB).
    // hypertile full_perm is src→dst = inverse(perm).
    std::vector<int32_t> ht_full_perm(rank);  // src→dst
    for (int i = 0; i < rank; i++) ht_full_perm[perm[i]] = i;

    auto r = build_task_for_shape(d, rank, rank, ht_full_perm.data(), perm.data(), 1024);
    if (r.skip) return -1.0f;

    int wr = r.shape.WR;

    // Pack shape
    auto pack_and_launch = [&](auto rank_tag) -> float {
        constexpr int R = decltype(rank_tag)::value;
        HypertileShape<R> shape = pack_shape<R>(r.shape);

        HypertileMem mem{};
        mem.src_base = 0;
        mem.dst_base = 0;
        mem.shape_idx = 0;

        int32_t prefix_sum[2] = {0, r.shape.num_tiles};

        // Upload
        HypertileShape<R>* d_shapes;
        HypertileMem* d_mems;
        int32_t* d_prefix;
        cudaMalloc(&d_shapes, sizeof(HypertileShape<R>));
        cudaMalloc(&d_mems, sizeof(HypertileMem));
        cudaMalloc(&d_prefix, 2 * sizeof(int32_t));
        cudaMemcpyAsync(d_shapes, &shape, sizeof(shape), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_mems, &mem, sizeof(mem), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_prefix, prefix_sum, sizeof(prefix_sum), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        auto result = gpu_bench([&]() {
            launch_hypertile_kernel<T, 1024, 256>(
                d_src, d_dst, d_shapes, d_mems, d_prefix,
                1, r.shape.num_tiles, wr, stream);
        }, warmup, repeat, stream);

        cudaFree(d_shapes);
        cudaFree(d_mems);
        cudaFree(d_prefix);
        return result.median_ms;
    };

    switch (wr) {
        case 2: return pack_and_launch(std::integral_constant<int,2>{});
        case 3: return pack_and_launch(std::integral_constant<int,3>{});
        case 4: return pack_and_launch(std::integral_constant<int,4>{});
        case 5: return pack_and_launch(std::integral_constant<int,5>{});
        case 6: return pack_and_launch(std::integral_constant<int,6>{});
        case 7: return pack_and_launch(std::integral_constant<int,7>{});
        case 8: return pack_and_launch(std::integral_constant<int,8>{});
        default: return pack_and_launch(std::integral_constant<int,8>{});
    }
}

// ---------------------------------------------------------------------------
// Hypertile batched benchmark (many blocks, same shape)
// ---------------------------------------------------------------------------

template <typename T>
static float bench_hypertile_batched(
    const std::vector<int64_t>& block_dims,
    const std::vector<int32_t>& perm,  // dst←src (cuTENSOR convention)
    int num_blocks,
    const T* d_src, T* d_dst,
    const std::vector<int64_t>& offsets,
    int warmup, int repeat, cudaStream_t stream)
{
    const int rank = static_cast<int>(block_dims.size());
    dim_t d[16];
    for (int i = 0; i < rank; i++) d[i] = static_cast<dim_t>(block_dims[i]);

    // perm is dst←src; hypertile full_perm is src→dst = inverse(perm)
    std::vector<int32_t> ht_full_perm(rank);
    for (int i = 0; i < rank; i++) ht_full_perm[perm[i]] = i;

    auto r = build_task_for_shape(d, rank, rank, ht_full_perm.data(), perm.data(), 1024);
    if (r.skip) return -1.0f;

    int wr = r.shape.WR;

    auto pack_and_launch = [&](auto rank_tag) -> float {
        constexpr int R = decltype(rank_tag)::value;
        HypertileShape<R> shape = pack_shape<R>(r.shape);

        // Build mems for all blocks
        std::vector<HypertileMem> mems(num_blocks);
        for (int b = 0; b < num_blocks; b++) {
            mems[b].src_base = offsets[b];
            mems[b].dst_base = offsets[b];
            mems[b].shape_idx = 0;
        }

        // Build prefix_sum
        std::vector<int32_t> prefix_sum(num_blocks + 1);
        prefix_sum[0] = 0;
        for (int b = 0; b < num_blocks; b++)
            prefix_sum[b + 1] = prefix_sum[b] + r.shape.num_tiles;
        int32_t total_tiles = prefix_sum[num_blocks];

        // Upload
        HypertileShape<R>* d_shapes;
        HypertileMem* d_mems;
        int32_t* d_prefix;
        cudaMalloc(&d_shapes, sizeof(HypertileShape<R>));
        cudaMalloc(&d_mems, num_blocks * sizeof(HypertileMem));
        cudaMalloc(&d_prefix, (num_blocks + 1) * sizeof(int32_t));
        cudaMemcpyAsync(d_shapes, &shape, sizeof(shape), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_mems, mems.data(), num_blocks * sizeof(HypertileMem),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_prefix, prefix_sum.data(), (num_blocks + 1) * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        auto result = gpu_bench([&]() {
            launch_hypertile_kernel<T, 1024, 256>(
                d_src, d_dst, d_shapes, d_mems, d_prefix,
                num_blocks, total_tiles, wr, stream);
        }, warmup, repeat, stream);

        cudaFree(d_shapes);
        cudaFree(d_mems);
        cudaFree(d_prefix);
        return result.median_ms;
    };

    switch (wr) {
        case 2: return pack_and_launch(std::integral_constant<int,2>{});
        case 3: return pack_and_launch(std::integral_constant<int,3>{});
        case 4: return pack_and_launch(std::integral_constant<int,4>{});
        case 5: return pack_and_launch(std::integral_constant<int,5>{});
        case 6: return pack_and_launch(std::integral_constant<int,6>{});
        case 7: return pack_and_launch(std::integral_constant<int,7>{});
        case 8: return pack_and_launch(std::integral_constant<int,8>{});
        default: return pack_and_launch(std::integral_constant<int,8>{});
    }
}

// ---------------------------------------------------------------------------
// Single large tensor benchmark
// ---------------------------------------------------------------------------

template <typename T>
static void run_single_bench(int warmup, int repeat, cudaStream_t stream, uint32_t base_seed) {
    printf("\n=== Single Tensor Transpose: cuTENSOR vs Hypertile ===\n");
    printf("%-12s %-22s %10s %10s %10s %10s %8s\n",
           "Perm", "Shape", "cuTENSOR", "Hypertile", "Speedup", "HT GB/s", "CT GB/s");
    printf("%-12s %-22s %10s %10s %10s %10s %8s\n",
           "----", "-----", "--------", "---------", "-------", "-------", "-------");

    int64_t volume_arr[] = {
        1L << 22,   // 4M
        1L << 24,   // 16M
        1L << 26,   // 64M (512 MB f32)
    };
    int rank_arr[] = {2, 3, 4, 5, 6};

    enum ShapeKind { Regular, Irregular };

    for (auto vol : volume_arr) {
        for (auto rank : rank_arr) {
            for (int kind = 0; kind < 2; kind++) {
                auto dims = (kind == Regular)
                    ? factor_dims(vol, rank)
                    : irregular_dims(vol, rank, base_seed);
                int64_t actual_vol = product(dims);

                // 4 GB total cap (2x for src+dst)
                if (actual_vol * static_cast<int64_t>(sizeof(T)) * 2 > 4LL * 1024 * 1024 * 1024)
                    continue;

                auto perm = make_random_perm(rank, base_seed);
                auto perm_str = perm_to_string(perm);
                auto shape_str = dims_to_string(dims);

                T *d_src, *d_dst;
                cudaError_t err = cudaMalloc(&d_src, actual_vol * sizeof(T));
                if (err != cudaSuccess) {
                    cudaGetLastError();
                    printf("%-12s %-22s %10s %10s %10s %10s %8s\n",
                           perm_str.c_str(), shape_str.c_str(), "OOM", "-", "-", "-", "-");
                    continue;
                }
                err = cudaMalloc(&d_dst, actual_vol * sizeof(T));
                if (err != cudaSuccess) {
                    cudaFree(d_src); cudaGetLastError();
                    printf("%-12s %-22s %10s %10s %10s %10s %8s\n",
                           perm_str.c_str(), shape_str.c_str(), "OOM", "-", "-", "-", "-");
                    continue;
                }
                cudaMemsetAsync(d_src, 0x3f, actual_vol * sizeof(T), stream);
                cudaStreamSynchronize(stream);

                float ct_ms = bench_cutensor_permute<T>(dims, perm, d_src, d_dst,
                                                         warmup, repeat, stream);
                float ht_ms = bench_hypertile_single<T>(dims, perm, d_src, d_dst,
                                                         warmup, repeat, stream);

                double bytes = static_cast<double>(actual_vol) * sizeof(T) * 2.0;

                if (ht_ms > 0) {
                    double ct_gbps = bytes / (ct_ms * 1e6);
                    double ht_gbps = bytes / (ht_ms * 1e6);
                    printf("%-12s %-22s %9.3fms %9.3fms %9.2fx %7.1f %7.1f\n",
                           perm_str.c_str(), shape_str.c_str(), ct_ms, ht_ms,
                           ct_ms / ht_ms, ht_gbps, ct_gbps);
                } else {
                    double ct_gbps = bytes / (ct_ms * 1e6);
                    printf("%-12s %-22s %9.3fms %9s %9s %7s %7.1f\n",
                           perm_str.c_str(), shape_str.c_str(), ct_ms, "skip", "-", "-", ct_gbps);
                }
                fflush(stdout);

                cudaFree(d_src);
                cudaFree(d_dst);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Batched small tensor benchmark
// ---------------------------------------------------------------------------

template <typename T>
static void run_batched_bench(int warmup, int repeat, cudaStream_t stream, uint32_t base_seed) {
    printf("\n=== Batched Transpose: cuTENSOR (loop) vs Hypertile (single launch) ===\n");
    printf("%-12s %-22s %8s %10s %10s %10s %10s %8s\n",
           "Perm", "BlockShape", "Blocks", "cuTENSOR", "Hypertile", "Speedup", "HT GB/s", "CT GB/s");
    printf("%-12s %-22s %8s %10s %10s %10s %10s %8s\n",
           "----", "----------", "------", "--------", "---------", "-------", "-------", "-------");

    int64_t block_vol_arr[] = {256, 1024, 4096, 16384, 65536};
    int num_blocks_arr[] = {10, 100, 1000};
    int rank_arr[] = {3, 5};

    for (auto block_vol : block_vol_arr) {
        for (auto rank : rank_arr) {
            for (int kind = 0; kind < 2; kind++) {
                auto block_dims = (kind == 0)
                    ? factor_dims(block_vol, rank)
                    : irregular_dims(block_vol, rank, base_seed);
                int64_t actual_block_vol = product(block_dims);
                auto perm = make_random_perm(rank, base_seed);
                auto perm_str = perm_to_string(perm);
                auto shape_str = dims_to_string(block_dims);

                for (auto num_blocks : num_blocks_arr) {
                    int64_t total_vol = actual_block_vol * num_blocks;
                    int64_t total_bytes = total_vol * static_cast<int64_t>(sizeof(T));

                    // Skip if > 4 GB
                    if (total_bytes > 4LL * 1024 * 1024 * 1024) {
                        printf("%-12s %-22s %8d %10s %10s %10s %10s %8s\n",
                               perm_str.c_str(), shape_str.c_str(), num_blocks,
                               "SKIP", "-", "-", "-", "-");
                        continue;
                    }

                    T *d_src, *d_dst;
                    cudaError_t err = cudaMalloc(&d_src, total_vol * sizeof(T));
                    if (err != cudaSuccess) {
                        cudaGetLastError();
                        printf("%-12s %-22s %8d %10s %10s %10s %10s %8s\n",
                               perm_str.c_str(), shape_str.c_str(), num_blocks,
                               "OOM", "-", "-", "-", "-");
                        continue;
                    }
                    err = cudaMalloc(&d_dst, total_vol * sizeof(T));
                    if (err != cudaSuccess) {
                        cudaFree(d_src); cudaGetLastError();
                        printf("%-12s %-22s %8d %10s %10s %10s %10s %8s\n",
                               perm_str.c_str(), shape_str.c_str(), num_blocks,
                               "OOM", "-", "-", "-", "-");
                        continue;
                    }
                    cudaMemsetAsync(d_src, 0x3f, total_vol * sizeof(T), stream);
                    cudaStreamSynchronize(stream);

                    std::vector<int64_t> offsets(num_blocks);
                    for (int b = 0; b < num_blocks; b++)
                        offsets[b] = static_cast<int64_t>(b) * actual_block_vol;

                    float ct_ms = bench_cutensor_batched<T>(block_dims, perm, num_blocks,
                                                             d_src, d_dst, offsets,
                                                             warmup, repeat, stream);
                    float ht_ms = bench_hypertile_batched<T>(block_dims, perm, num_blocks,
                                                              d_src, d_dst, offsets,
                                                              warmup, repeat, stream);

                    // Skip if both times are < 0.1ms (noise-dominated)
                    float max_ms = std::max(ct_ms, ht_ms > 0 ? ht_ms : ct_ms);
                    if (max_ms < 0.1f) {
                        cudaFree(d_src);
                        cudaFree(d_dst);
                        continue;
                    }

                    double bytes = static_cast<double>(total_vol) * sizeof(T) * 2.0;

                    if (ht_ms > 0) {
                        double ct_gbps = bytes / (ct_ms * 1e6);
                        double ht_gbps = bytes / (ht_ms * 1e6);
                        printf("%-12s %-22s %8d %9.3fms %9.3fms %9.2fx %7.1f %7.1f\n",
                               perm_str.c_str(), shape_str.c_str(), num_blocks,
                               ct_ms, ht_ms, ct_ms / ht_ms, ht_gbps, ct_gbps);
                    } else {
                        double ct_gbps = bytes / (ct_ms * 1e6);
                        printf("%-12s %-22s %8d %9.3fms %9s %9s %7s %7.1f\n",
                               perm_str.c_str(), shape_str.c_str(), num_blocks,
                               ct_ms, "skip", "-", "-", ct_gbps);
                    }

                    fflush(stdout);
                    cudaFree(d_src);
                    cudaFree(d_dst);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mixed block sizes benchmark (realistic: blocks with varying shapes)
// ---------------------------------------------------------------------------

template <typename T>
static void run_mixed_bench(int warmup, int repeat, cudaStream_t stream) {
    printf("\n=== Mixed Block Sizes: cuTENSOR (per-shape plan) vs Hypertile ===\n");
    printf("%-8s %6s %8s %6s %10s %10s %10s\n",
           "Perm", "AvgVol", "Blocks", "Rank", "cuTENSOR", "Hypertile", "Speedup");
    printf("%-8s %6s %8s %6s %10s %10s %10s\n",
           "----", "------", "------", "----", "--------", "---------", "-------");

    // Mix: 3 block shapes with different sizes
    int num_blocks_arr[] = {10, 100, 1000};
    int rank_arr[] = {3, 5};

    for (auto rank : rank_arr) {
        auto perm = make_random_perm(rank, 42);

        // Define 3 block shapes
        auto dims_small  = factor_dims(256, rank);
        auto dims_medium = factor_dims(4096, rank);
        auto dims_large  = factor_dims(65536, rank);

        std::vector<std::vector<int64_t>> shape_pool = {dims_small, dims_medium, dims_large};

        for (auto total_blocks : num_blocks_arr) {
            // Assign shapes cyclically: 1/3 each
            std::vector<int> shape_idx(total_blocks);
            std::vector<int64_t> block_vols(total_blocks);
            for (int b = 0; b < total_blocks; b++) {
                shape_idx[b] = b % 3;
                block_vols[b] = product(shape_pool[shape_idx[b]]);
            }

            // Compute offsets (packed layout)
            std::vector<int64_t> offsets(total_blocks);
            int64_t total_vol = 0;
            for (int b = 0; b < total_blocks; b++) {
                offsets[b] = total_vol;
                total_vol += block_vols[b];
            }

            int64_t total_bytes = total_vol * static_cast<int64_t>(sizeof(T));
            if (total_bytes > 4LL * 1024 * 1024 * 1024) {
                printf("%-8s %6s %8d %6d %10s %10s %10s\n",
                       "reverse", "mixed", total_blocks, rank, "SKIP", "-", "-");
                continue;
            }

            T *d_src, *d_dst;
            if (cudaMalloc(&d_src, total_vol * sizeof(T)) != cudaSuccess) {
                cudaGetLastError();
                printf("%-8s %6s %8d %6d %10s %10s %10s\n",
                       "reverse", "mixed", total_blocks, rank, "OOM", "-", "-");
                continue;
            }
            if (cudaMalloc(&d_dst, total_vol * sizeof(T)) != cudaSuccess) {
                cudaFree(d_src); cudaGetLastError();
                printf("%-8s %6s %8d %6d %10s %10s %10s\n",
                       "reverse", "mixed", total_blocks, rank, "OOM", "-", "-");
                continue;
            }
            cudaMemsetAsync(d_src, 0x3f, total_vol * sizeof(T), stream);
            cudaStreamSynchronize(stream);

            // --- cuTENSOR: one plan per shape, loop over blocks ---
            // Group blocks by shape, create one plan per group
            struct ShapePlan {
                cutensorPlan_t plan{};
                T alpha = T(1);
            };

            cutensorHandle_t handle = get_handle();
            std::vector<ShapePlan> plans(3);

            for (int si = 0; si < 3; si++) {
                auto& bd = shape_pool[si];
                uint32_t r = static_cast<uint32_t>(rank);
                std::vector<int64_t> src_s(r);
                src_s[r-1] = 1;
                for (int d = static_cast<int>(r)-2; d >= 0; --d)
                    src_s[d] = src_s[d+1] * bd[d+1];
                std::vector<int64_t> dst_d(r);
                for (uint32_t d = 0; d < r; d++) dst_d[d] = bd[perm[d]];
                std::vector<int64_t> dst_s(r);
                dst_s[r-1] = 1;
                for (int d = static_cast<int>(r)-2; d >= 0; --d)
                    dst_s[d] = dst_s[d+1] * dst_d[d+1];

                std::vector<int32_t> mA(r), mB(r);
                for (uint32_t d = 0; d < r; d++) mA[d] = static_cast<int32_t>(d);
                for (uint32_t d = 0; d < r; d++) mB[d] = perm[d];

                cudaDataType_t dtype = std::is_same<T,float>::value ? CUDA_R_32F : CUDA_R_64F;
                auto comp = std::is_same<T,float>::value
                    ? CUTENSOR_COMPUTE_DESC_32F : CUTENSOR_COMPUTE_DESC_64F;

                const uint32_t align = static_cast<uint32_t>(sizeof(T));
                cutensorTensorDescriptor_t dA{}, dB{};
                check_cutensor(cutensorCreateTensorDescriptor(handle, &dA, r,
                    bd.data(), src_s.data(), dtype, align), "dA");
                check_cutensor(cutensorCreateTensorDescriptor(handle, &dB, r,
                    dst_d.data(), dst_s.data(), dtype, align), "dB");

                cutensorOperationDescriptor_t op{};
                check_cutensor(cutensorCreatePermutation(handle, &op,
                    dA, mA.data(), CUTENSOR_OP_IDENTITY,
                    dB, mB.data(), comp), "createPerm");

                cutensorPlanPreference_t pref{};
                check_cutensor(cutensorCreatePlanPreference(handle, &pref,
                    CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE), "pref");

                uint64_t ws = 0;
                cutensorEstimateWorkspaceSize(handle, op, pref,
                    CUTENSOR_WORKSPACE_DEFAULT, &ws);
                check_cutensor(cutensorCreatePlan(handle, &plans[si].plan,
                    op, pref, ws), "plan");

                cutensorDestroyPlanPreference(pref);
                cutensorDestroyOperationDescriptor(op);
                cutensorDestroyTensorDescriptor(dB);
                cutensorDestroyTensorDescriptor(dA);
            }

            int ct_bench_blocks = std::min(total_blocks, 1000);
            float ct_scale = static_cast<float>(total_blocks) / static_cast<float>(ct_bench_blocks);

            auto ct_result = gpu_bench([&]() {
                for (int b = 0; b < ct_bench_blocks; b++) {
                    int si = shape_idx[b];
                    cutensorPermute(handle, plans[si].plan, &plans[si].alpha,
                                    d_src + offsets[b], d_dst + offsets[b], stream);
                }
            }, warmup, repeat, stream);
            ct_result.median_ms *= ct_scale;

            for (int si = 0; si < 3; si++)
                cutensorDestroyPlan(plans[si].plan);

            // --- Hypertile: build per-shape tasks, single kernel launch per shape ---
            // Group blocks by shape for hypertile
            std::vector<std::vector<int>> groups(3);
            for (int b = 0; b < total_blocks; b++)
                groups[shape_idx[b]].push_back(b);

            // For each shape, build task and prepare GPU data
            struct HTSliceData {
                void* d_shapes;
                void* d_mems;
                int32_t* d_prefix;
                int num_tasks;
                int32_t total_tiles;
                int wr;
            };
            std::vector<HTSliceData> ht_slices;

            std::vector<int32_t> inv_perm(rank);
            for (int i = 0; i < rank; i++) inv_perm[perm[i]] = i;

            auto build_ht_slice = [&](int si, auto rank_tag) {
                constexpr int R = decltype(rank_tag)::value;
                auto& bd = shape_pool[si];
                dim_t d[16];
                for (int i = 0; i < rank; i++) d[i] = static_cast<dim_t>(bd[i]);

                auto r = build_task_for_shape(d, rank, rank, inv_perm.data(), perm.data(), 1024);
                if (r.skip || groups[si].empty()) return;

                HypertileShape<R> shape = pack_shape<R>(r.shape);
                int nb = static_cast<int>(groups[si].size());

                std::vector<HypertileMem> mems(nb);
                for (int i = 0; i < nb; i++) {
                    mems[i].src_base = offsets[groups[si][i]];
                    mems[i].dst_base = offsets[groups[si][i]];
                    mems[i].shape_idx = 0;
                }

                std::vector<int32_t> ps(nb + 1);
                ps[0] = 0;
                for (int i = 0; i < nb; i++)
                    ps[i+1] = ps[i] + r.shape.num_tiles;

                HTSliceData sl{};
                sl.num_tasks = nb;
                sl.total_tiles = ps[nb];
                sl.wr = r.shape.WR;

                cudaMalloc(&sl.d_shapes, sizeof(HypertileShape<R>));
                cudaMalloc(&sl.d_mems, nb * sizeof(HypertileMem));
                cudaMalloc(&sl.d_prefix, (nb + 1) * sizeof(int32_t));
                cudaMemcpyAsync(sl.d_shapes, &shape, sizeof(shape), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(sl.d_mems, mems.data(), nb * sizeof(HypertileMem),
                                cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(sl.d_prefix, ps.data(), (nb + 1) * sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream);
                ht_slices.push_back(sl);
            };

            for (int si = 0; si < 3; si++) {
                if (groups[si].empty()) continue;
                dim_t dd[16];
                for (int i = 0; i < rank; i++) dd[i] = static_cast<dim_t>(shape_pool[si][i]);
                auto r = build_task_for_shape(dd, rank, rank, inv_perm.data(), perm.data(), 1024);
                if (r.skip) continue;
                int wr = r.shape.WR;
                switch (wr) {
                    case 2: build_ht_slice(si, std::integral_constant<int,2>{}); break;
                    case 3: build_ht_slice(si, std::integral_constant<int,3>{}); break;
                    case 4: build_ht_slice(si, std::integral_constant<int,4>{}); break;
                    case 5: build_ht_slice(si, std::integral_constant<int,5>{}); break;
                    case 6: build_ht_slice(si, std::integral_constant<int,6>{}); break;
                    case 7: build_ht_slice(si, std::integral_constant<int,7>{}); break;
                    case 8: build_ht_slice(si, std::integral_constant<int,8>{}); break;
                    default: break;
                }
            }
            cudaStreamSynchronize(stream);

            auto ht_result = gpu_bench([&]() {
                for (auto& sl : ht_slices) {
                    launch_hypertile_kernel<T, 1024, 256>(
                        d_src, d_dst, sl.d_shapes, sl.d_mems, sl.d_prefix,
                        sl.num_tasks, sl.total_tiles, sl.wr, stream);
                }
            }, warmup, repeat, stream);

            for (auto& sl : ht_slices) {
                cudaFree(sl.d_shapes);
                cudaFree(sl.d_mems);
                cudaFree(sl.d_prefix);
            }

            int64_t avg_vol = total_vol / total_blocks;
            float speedup = ct_result.median_ms / ht_result.median_ms;
            printf("%-8s %6ld %8d %6d %9.3fms %9.3fms %9.2fx\n",
                   "reverse", avg_vol, total_blocks, rank,
                   ct_result.median_ms, ht_result.median_ms, speedup);

            cudaFree(d_src);
            cudaFree(d_dst);
        }
    }
}

// ---------------------------------------------------------------------------
// Batch sweep benchmark: speedup vs batch size at constant total volume
// ---------------------------------------------------------------------------

// Helper: build cuTENSOR plan for a given block shape (pre-timing setup).
template <typename T>
struct CTPlanHandle {
    cutensorPlan_t plan{};
    T alpha = T(1);

    ~CTPlanHandle() { if (plan) cutensorDestroyPlan(plan); }
    CTPlanHandle() = default;
    CTPlanHandle(const CTPlanHandle&) = delete;
    CTPlanHandle& operator=(const CTPlanHandle&) = delete;
    CTPlanHandle(CTPlanHandle&& o) : plan(o.plan), alpha(o.alpha) { o.plan = {}; }
    CTPlanHandle& operator=(CTPlanHandle&& o) { plan = o.plan; alpha = o.alpha; o.plan = {}; return *this; }
};

template <typename T>
static CTPlanHandle<T> make_ct_plan(
    const std::vector<int64_t>& block_dims,
    const std::vector<int32_t>& perm)
{
    cutensorHandle_t handle = get_handle();
    const uint32_t rank = static_cast<uint32_t>(block_dims.size());

    std::vector<int64_t> src_strides(rank);
    src_strides[rank - 1] = 1;
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d)
        src_strides[d] = src_strides[d + 1] * block_dims[d + 1];

    std::vector<int64_t> dst_dims(rank);
    for (uint32_t d = 0; d < rank; d++) dst_dims[d] = block_dims[perm[d]];
    std::vector<int64_t> dst_strides(rank);
    dst_strides[rank - 1] = 1;
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d)
        dst_strides[d] = dst_strides[d + 1] * dst_dims[d + 1];

    std::vector<int32_t> modeA(rank), modeB(rank);
    for (uint32_t d = 0; d < rank; d++) modeA[d] = static_cast<int32_t>(d);
    for (uint32_t d = 0; d < rank; d++) modeB[d] = perm[d];

    cudaDataType_t dtype = std::is_same<T, float>::value ? CUDA_R_32F : CUDA_R_64F;
    auto compute = std::is_same<T, float>::value
        ? CUTENSOR_COMPUTE_DESC_32F : CUTENSOR_COMPUTE_DESC_64F;
    const uint32_t align = static_cast<uint32_t>(sizeof(T));

    cutensorTensorDescriptor_t descA{}, descB{};
    check_cutensor(cutensorCreateTensorDescriptor(handle, &descA, rank,
        block_dims.data(), src_strides.data(), dtype, align), "descA");
    check_cutensor(cutensorCreateTensorDescriptor(handle, &descB, rank,
        dst_dims.data(), dst_strides.data(), dtype, align), "descB");

    cutensorOperationDescriptor_t opDesc{};
    check_cutensor(cutensorCreatePermutation(handle, &opDesc,
        descA, modeA.data(), CUTENSOR_OP_IDENTITY,
        descB, modeB.data(), compute), "createPermutation");

    cutensorPlanPreference_t pref{};
    check_cutensor(cutensorCreatePlanPreference(handle, &pref,
        CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE), "createPlanPref");

    uint64_t ws_size = 0;
    cutensorEstimateWorkspaceSize(handle, opDesc, pref,
        CUTENSOR_WORKSPACE_DEFAULT, &ws_size);

    CTPlanHandle<T> result;
    check_cutensor(cutensorCreatePlan(handle, &result.plan, opDesc, pref, ws_size),
                   "createPlan");

    cutensorDestroyPlanPreference(pref);
    cutensorDestroyOperationDescriptor(opDesc);
    cutensorDestroyTensorDescriptor(descB);
    cutensorDestroyTensorDescriptor(descA);
    return result;
}

// Helper: pre-built hypertile GPU state for a shape group.
struct HTGroupGPU {
    void* d_shapes = nullptr;
    void* d_mems = nullptr;
    int32_t* d_prefix = nullptr;
    int num_tasks = 0;
    int32_t total_tiles = 0;
    int wr = 0;

    ~HTGroupGPU() {
        if (d_shapes) cudaFree(d_shapes);
        if (d_mems) cudaFree(d_mems);
        if (d_prefix) cudaFree(d_prefix);
    }
    HTGroupGPU() = default;
    HTGroupGPU(const HTGroupGPU&) = delete;
    HTGroupGPU& operator=(const HTGroupGPU&) = delete;
    HTGroupGPU(HTGroupGPU&& o) : d_shapes(o.d_shapes), d_mems(o.d_mems),
        d_prefix(o.d_prefix), num_tasks(o.num_tasks),
        total_tiles(o.total_tiles), wr(o.wr)
    { o.d_shapes = o.d_mems = nullptr; o.d_prefix = nullptr; }
    HTGroupGPU& operator=(HTGroupGPU&& o) {
        d_shapes = o.d_shapes; d_mems = o.d_mems; d_prefix = o.d_prefix;
        num_tasks = o.num_tasks; total_tiles = o.total_tiles; wr = o.wr;
        o.d_shapes = o.d_mems = nullptr; o.d_prefix = nullptr;
        return *this;
    }
};

// Build and upload hypertile GPU state for a group of blocks sharing one shape.
template <typename T>
static HTGroupGPU build_ht_group(
    const std::vector<int64_t>& block_dims,
    const std::vector<int32_t>& perm,
    const std::vector<int64_t>& group_offsets,  // offsets into the global buffer
    cudaStream_t stream)
{
    const int rank = static_cast<int>(block_dims.size());
    dim_t d[16];
    for (int i = 0; i < rank; i++) d[i] = static_cast<dim_t>(block_dims[i]);

    std::vector<int32_t> inv_perm(rank);
    for (int i = 0; i < rank; i++) inv_perm[perm[i]] = i;

    auto r = build_task_for_shape(d, rank, rank, inv_perm.data(), perm.data(), 1024);
    if (r.skip) return {};

    int wr = r.shape.WR;
    int nb = static_cast<int>(group_offsets.size());

    auto do_upload = [&](auto rank_tag) -> HTGroupGPU {
        constexpr int R = decltype(rank_tag)::value;
        HypertileShape<R> shape = pack_shape<R>(r.shape);

        std::vector<HypertileMem> mems(nb);
        for (int i = 0; i < nb; i++) {
            mems[i].src_base = group_offsets[i];
            mems[i].dst_base = group_offsets[i];
            mems[i].shape_idx = 0;
        }

        std::vector<int32_t> ps(nb + 1);
        ps[0] = 0;
        for (int i = 0; i < nb; i++)
            ps[i + 1] = ps[i] + r.shape.num_tiles;

        HTGroupGPU g;
        g.num_tasks = nb;
        g.total_tiles = ps[nb];
        g.wr = wr;

        cudaMalloc(&g.d_shapes, sizeof(HypertileShape<R>));
        cudaMalloc(&g.d_mems, nb * sizeof(HypertileMem));
        cudaMalloc(&g.d_prefix, (nb + 1) * sizeof(int32_t));
        cudaMemcpyAsync(g.d_shapes, &shape, sizeof(shape), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(g.d_mems, mems.data(), nb * sizeof(HypertileMem),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(g.d_prefix, ps.data(), (nb + 1) * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
        return g;
    };

    switch (wr) {
        case 2: return do_upload(std::integral_constant<int,2>{});
        case 3: return do_upload(std::integral_constant<int,3>{});
        case 4: return do_upload(std::integral_constant<int,4>{});
        case 5: return do_upload(std::integral_constant<int,5>{});
        case 6: return do_upload(std::integral_constant<int,6>{});
        case 7: return do_upload(std::integral_constant<int,7>{});
        case 8: return do_upload(std::integral_constant<int,8>{});
        default: return do_upload(std::integral_constant<int,8>{});
    }
}

template <typename T>
static void run_batch_sweep_bench(int warmup, int repeat, cudaStream_t stream,
                                  uint32_t base_seed, int64_t total_vol) {
    printf("\n=== Batch Sweep Transpose: speedup vs batch size (rank=5, total_vol=%ld) ===\n",
           (long)total_vol);
    printf("%-8s %8s %10s %10s %10s %10s\n",
           "Mode", "Blocks", "BlockVol", "cuTENSOR", "Hypertile", "Speedup");
    printf("%-8s %8s %10s %10s %10s %10s\n",
           "----", "------", "--------", "--------", "---------", "-------");

    const int rank = 5;
    auto perm = make_random_perm(rank, base_seed);
    cutensorHandle_t handle = get_handle();

    int batch_sizes[] = {1, 5, 10, 50, 100, 500, 1000};

    for (auto num_blocks : batch_sizes) {
        int64_t block_vol = total_vol / num_blocks;
        if (block_vol < 32) continue;  // too small to be meaningful

        // ── Fixed mode: all blocks same shape ──
        {
            auto block_dims = factor_dims(block_vol, rank);
            int64_t actual_block_vol = product(block_dims);
            int64_t actual_total = actual_block_vol * num_blocks;

            T *d_src = nullptr, *d_dst = nullptr;
            bool fixed_ok = true;
            if (cudaMalloc(&d_src, actual_total * sizeof(T)) != cudaSuccess) {
                cudaGetLastError();
                printf("%-8s %8d %10ld %10s %10s %10s\n",
                       "fixed", num_blocks, (long)actual_block_vol, "OOM", "-", "-");
                fixed_ok = false;
            }
            if (fixed_ok && cudaMalloc(&d_dst, actual_total * sizeof(T)) != cudaSuccess) {
                cudaFree(d_src); cudaGetLastError();
                printf("%-8s %8d %10ld %10s %10s %10s\n",
                       "fixed", num_blocks, (long)actual_block_vol, "OOM", "-", "-");
                fixed_ok = false;
            }

            if (fixed_ok) {
                cudaMemsetAsync(d_src, 0x3f, actual_total * sizeof(T), stream);
                cudaStreamSynchronize(stream);

                // Build offsets
                std::vector<int64_t> offsets(num_blocks);
                for (int b = 0; b < num_blocks; b++)
                    offsets[b] = static_cast<int64_t>(b) * actual_block_vol;

                // Pre-build cuTENSOR plan (outside timing)
                auto ct_plan = make_ct_plan<T>(block_dims, perm);

                int ct_bench_blocks = std::min(num_blocks, 1000);
                float ct_scale = static_cast<float>(num_blocks) / static_cast<float>(ct_bench_blocks);

                auto ct_result = gpu_bench([&]() {
                    for (int b = 0; b < ct_bench_blocks; b++) {
                        cutensorPermute(handle, ct_plan.plan, &ct_plan.alpha,
                                        d_src + offsets[b], d_dst + offsets[b], stream);
                    }
                }, warmup, repeat, stream);
                float ct_ms = ct_result.median_ms * ct_scale;

                // Pre-build hypertile GPU state (outside timing)
                auto ht_group = build_ht_group<T>(block_dims, perm, offsets, stream);
                cudaStreamSynchronize(stream);

                float ht_ms = -1.0f;
                if (ht_group.num_tasks > 0) {
                    auto ht_result = gpu_bench([&]() {
                        launch_hypertile_kernel<T, 1024, 256>(
                            d_src, d_dst, ht_group.d_shapes, ht_group.d_mems,
                            ht_group.d_prefix, ht_group.num_tasks,
                            ht_group.total_tiles, ht_group.wr, stream);
                    }, warmup, repeat, stream);
                    ht_ms = ht_result.median_ms;
                }

                if (ht_ms > 0) {
                    printf("%-8s %8d %10ld %9.3fms %9.3fms %9.2fx\n",
                           "fixed", num_blocks, (long)actual_block_vol,
                           ct_ms, ht_ms, ct_ms / ht_ms);
                } else {
                    printf("%-8s %8d %10ld %9.3fms %9s %9s\n",
                           "fixed", num_blocks, (long)actual_block_vol,
                           ct_ms, "skip", "-");
                }
                fflush(stdout);

                cudaFree(d_src);
                cudaFree(d_dst);
            }
        }

        // ── Mixed mode: ceil(num_blocks/10) distinct shapes ──
        {
            int num_shapes = (num_blocks + 9) / 10;
            if (num_shapes < 1) num_shapes = 1;

            // Generate distinct shapes targeting block_vol.
            // Start from factor_dims, then perturb: double one dim and halve another
            // to create shapes with the same volume but different aspect ratios.
            std::vector<std::vector<int64_t>> shapes(num_shapes);
            auto base_dims = factor_dims(block_vol, rank);
            shapes[0] = base_dims;
            for (int s = 1; s < num_shapes; s++) {
                shapes[s] = base_dims;
                // Pick two dims to perturb: multiply dim[a] by 2, divide dim[b] by 2
                int a = s % rank;
                int b = (s + 1) % rank;
                if (a == b) b = (b + 1) % rank;
                shapes[s][a] *= 2;
                shapes[s][b] = std::max(int64_t(2), shapes[s][b] / 2);
            }

            // Assign shapes cyclically
            std::vector<int> shape_idx(num_blocks);
            std::vector<int64_t> block_vols(num_blocks);
            for (int b = 0; b < num_blocks; b++) {
                shape_idx[b] = b % num_shapes;
                block_vols[b] = product(shapes[shape_idx[b]]);
            }

            // Compute offsets
            std::vector<int64_t> offsets(num_blocks);
            int64_t actual_total = 0;
            for (int b = 0; b < num_blocks; b++) {
                offsets[b] = actual_total;
                actual_total += block_vols[b];
            }

            int64_t avg_block_vol = actual_total / num_blocks;

            T *d_src = nullptr, *d_dst = nullptr;
            if (cudaMalloc(&d_src, actual_total * sizeof(T)) != cudaSuccess) {
                cudaGetLastError();
                printf("%-8s %8d %10ld %10s %10s %10s\n",
                       "mixed", num_blocks, (long)avg_block_vol, "OOM", "-", "-");
                continue;
            }
            if (cudaMalloc(&d_dst, actual_total * sizeof(T)) != cudaSuccess) {
                cudaFree(d_src); cudaGetLastError();
                printf("%-8s %8d %10ld %10s %10s %10s\n",
                       "mixed", num_blocks, (long)avg_block_vol, "OOM", "-", "-");
                continue;
            }
            cudaMemsetAsync(d_src, 0x3f, actual_total * sizeof(T), stream);
            cudaStreamSynchronize(stream);

            // Pre-build cuTENSOR plans for each shape (outside timing)
            std::vector<CTPlanHandle<T>> ct_plans;
            ct_plans.reserve(num_shapes);
            for (int s = 0; s < num_shapes; s++)
                ct_plans.push_back(make_ct_plan<T>(shapes[s], perm));

            int ct_bench_blocks = std::min(num_blocks, 1000);
            float ct_scale = static_cast<float>(num_blocks) / static_cast<float>(ct_bench_blocks);

            auto ct_result = gpu_bench([&]() {
                for (int b = 0; b < ct_bench_blocks; b++) {
                    int si = shape_idx[b];
                    cutensorPermute(handle, ct_plans[si].plan, &ct_plans[si].alpha,
                                    d_src + offsets[b], d_dst + offsets[b], stream);
                }
            }, warmup, repeat, stream);
            float ct_ms = ct_result.median_ms * ct_scale;

            // Pre-build hypertile GPU state per shape group (outside timing)
            // Group block offsets by shape
            std::vector<std::vector<int64_t>> group_offsets(num_shapes);
            for (int b = 0; b < num_blocks; b++)
                group_offsets[shape_idx[b]].push_back(offsets[b]);

            std::vector<HTGroupGPU> ht_groups;
            for (int s = 0; s < num_shapes; s++) {
                if (group_offsets[s].empty()) continue;
                auto g = build_ht_group<T>(shapes[s], perm, group_offsets[s], stream);
                if (g.num_tasks > 0)
                    ht_groups.push_back(std::move(g));
            }
            cudaStreamSynchronize(stream);

            float ht_ms = -1.0f;
            if (!ht_groups.empty()) {
                auto ht_result = gpu_bench([&]() {
                    for (auto& g : ht_groups) {
                        launch_hypertile_kernel<T, 1024, 256>(
                            d_src, d_dst, g.d_shapes, g.d_mems, g.d_prefix,
                            g.num_tasks, g.total_tiles, g.wr, stream);
                    }
                }, warmup, repeat, stream);
                ht_ms = ht_result.median_ms;
            }

            if (ht_ms > 0) {
                printf("%-8s %8d %10ld %9.3fms %9.3fms %9.2fx\n",
                       "mixed", num_blocks, (long)avg_block_vol,
                       ct_ms, ht_ms, ct_ms / ht_ms);
            } else {
                printf("%-8s %8d %10ld %9.3fms %9s %9s\n",
                       "mixed", num_blocks, (long)avg_block_vol,
                       ct_ms, "skip", "-");
            }
            fflush(stdout);

            cudaFree(d_src);
            cudaFree(d_dst);
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

// Parse "AxBxC" into vector of dims.
static std::vector<int64_t> parse_shape(const std::string& s) {
    std::vector<int64_t> dims;
    std::string tok;
    for (char c : s) {
        if (c == 'x' || c == 'X') {
            if (!tok.empty()) { dims.push_back(std::atol(tok.c_str())); tok.clear(); }
        } else {
            tok += c;
        }
    }
    if (!tok.empty()) dims.push_back(std::atol(tok.c_str()));
    return dims;
}

// Parse "1,2,0" into vector of int32.
static std::vector<int32_t> parse_perm(const std::string& s) {
    std::vector<int32_t> perm;
    std::string tok;
    for (char c : s) {
        if (c == ',') {
            perm.push_back(std::atoi(tok.c_str())); tok.clear();
        } else {
            tok += c;
        }
    }
    if (!tok.empty()) perm.push_back(std::atoi(tok.c_str()));
    return perm;
}

// Single-case reproduce mode: verify + bench one specific (shape, perm).
template <typename T>
static void run_single_case(const std::vector<int64_t>& dims,
                            const std::vector<int32_t>& perm,
                            int warmup, int repeat, cudaStream_t stream) {
    int64_t vol = product(dims);
    auto shape_str = dims_to_string(dims);
    auto perm_str = perm_to_string(perm);

    printf("Shape: %s  Perm: %s  Volume: %ld  (%.1f MB)\n",
           shape_str.c_str(), perm_str.c_str(), (long)vol,
           vol * sizeof(T) / 1e6);

    T *d_src, *d_dst, *d_ct_out;
    check_cuda(cudaMalloc(&d_src, vol * sizeof(T)), "malloc src");
    check_cuda(cudaMalloc(&d_dst, vol * sizeof(T)), "malloc dst");
    check_cuda(cudaMalloc(&d_ct_out, vol * sizeof(T)), "malloc ct_out");

    // Sequential fill
    std::vector<T> h_src(vol);
    for (int64_t i = 0; i < vol; i++) h_src[i] = static_cast<T>(i % 100000 + 1);
    cudaMemcpy(d_src, h_src.data(), vol * sizeof(T), cudaMemcpyHostToDevice);

    // cuTENSOR
    cudaMemset(d_ct_out, 0, vol * sizeof(T));
    bench_cutensor_permute<T>(dims, perm, d_src, d_ct_out, 0, 1, stream);
    cudaStreamSynchronize(stream);

    // Hypertile
    cudaMemset(d_dst, 0, vol * sizeof(T));
    float ht_check = bench_hypertile_single<T>(dims, perm, d_src, d_dst, 0, 1, stream);
    cudaStreamSynchronize(stream);

    if (ht_check < 0) {
        printf("Hypertile: SKIP (identity after squash)\n");
    } else {
        int err = verify_against_cutensor(d_ct_out, d_dst, vol, shape_str.c_str());
        if (err)
            printf("VERIFY: FAIL (%d / %ld mismatches)\n", err, (long)vol);
        else
            printf("VERIFY: PASS\n");
    }

    // Timing
    cudaMemsetAsync(d_src, 0x3f, vol * sizeof(T), stream);
    cudaStreamSynchronize(stream);

    float ct_ms = bench_cutensor_permute<T>(dims, perm, d_src, d_dst, warmup, repeat, stream);
    float ht_ms = bench_hypertile_single<T>(dims, perm, d_src, d_dst, warmup, repeat, stream);
    double bytes = static_cast<double>(vol) * sizeof(T) * 2.0;
    double ct_gbps = bytes / (ct_ms * 1e6);
    double ht_gbps = (ht_ms > 0) ? bytes / (ht_ms * 1e6) : 0;

    printf("cuTENSOR: %.3fms (%.1f GB/s)\n", ct_ms, ct_gbps);
    if (ht_ms > 0)
        printf("Hypertile: %.3fms (%.1f GB/s)  Speedup: %.2fx\n",
               ht_ms, ht_gbps, ct_ms / ht_ms);

    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_ct_out);
}

// ---------------------------------------------------------------------------
// EXP-1b(a): Per-shape transpose bandwidth with real DMRG shapes
// ---------------------------------------------------------------------------

template <typename T>
static void run_exp1b_a(int warmup, int repeat, cudaStream_t stream) {
    printf("\n=== EXP-1b(a): Per-Shape Transpose Bandwidth ===\n");
    printf("%-22s %-18s %3s %6s %10s %10s %10s %8s %8s %8s\n",
           "Shape", "Perm", "WR", "Blocks", "TotalVol", "cuTENSOR", "Hypertile",
           "Speedup", "HT GB/s", "CT GB/s");
    printf("%-22s %-18s %3s %6s %10s %10s %10s %8s %8s %8s\n",
           "-----", "----", "--", "------", "--------", "--------", "---------",
           "-------", "-------", "-------");

    cutensorHandle_t handle = get_handle();

    struct ShapeCase {
        std::vector<int64_t> dims;
        std::vector<std::vector<int32_t>> perms;
    };

    // 4 shapes from real DMRG blocks, chosen for high dimension asymmetry
    std::vector<ShapeCase> cases = {
        // Rank 4 near-square (from nf_E2 A blocks)
        {{373, 6, 200, 2}, {{1,0,2,3}, {0,2,1,3}, {2,0,1,3}, {3,0,2,1}}},
        // Rank 4 thin-long (from nf_E1/S2 blocks)
        {{430, 2, 416, 8}, {{1,0,2,3}, {0,2,1,3}, {2,0,1,3}, {3,0,2,1}}},
        // Rank 6 medium (from nfparity_S3 A blocks)
        {{65, 15, 16, 6, 6, 2}, {{2,0,3,4,1,5}, {1,4,0,2,3,5}, {5,4,3,2,1,0}, {0,2,4,1,3,5}}},
        // Rank 6 large (from nfparity_S3 A blocks)
        {{802, 801, 6, 6, 2, 1}, {{2,0,3,4,1,5}, {1,4,0,2,3,5}, {5,4,3,2,1,0}, {0,2,4,1,3,5}}},
    };

    // Target ~500 MB total per case
    const int64_t target_bytes = 500LL * 1024 * 1024;

    for (auto& sc : cases) {
        int64_t block_vol = product(sc.dims);
        int rank = static_cast<int>(sc.dims.size());
        auto shape_str = dims_to_string(sc.dims);

        for (auto& perm : sc.perms) {
            auto perm_str = perm_to_string(perm);

            // Compute WR for this (shape, perm)
            dim_t dd[16];
            for (int i = 0; i < rank; i++) dd[i] = static_cast<dim_t>(sc.dims[i]);
            std::vector<int32_t> inv_perm(rank);
            for (int i = 0; i < rank; i++) inv_perm[perm[i]] = i;
            auto task = build_task_for_shape(dd, rank, rank, inv_perm.data(), perm.data(), 1024);
            if (task.skip) {
                printf("%-22s %-18s %3s %6s %10s %10s %10s %8s %8s %8s\n",
                       shape_str.c_str(), perm_str.c_str(), "-", "-",
                       "SKIP", "SKIP", "SKIP", "-", "-", "-");
                continue;
            }
            int wr = task.shape.WR;

            int num_blocks = std::max(1, static_cast<int>(target_bytes / (block_vol * static_cast<int64_t>(sizeof(T)))));
            int64_t total_vol = static_cast<int64_t>(num_blocks) * block_vol;
            int64_t total_bytes_alloc = total_vol * static_cast<int64_t>(sizeof(T));

            // Cap at 4 GB
            if (total_bytes_alloc > 4LL * 1024 * 1024 * 1024) {
                num_blocks = static_cast<int>(4LL * 1024 * 1024 * 1024 / (block_vol * static_cast<int64_t>(sizeof(T))));
                total_vol = static_cast<int64_t>(num_blocks) * block_vol;
                total_bytes_alloc = total_vol * static_cast<int64_t>(sizeof(T));
            }

            T *d_src = nullptr, *d_dst = nullptr;
            if (cudaMalloc(&d_src, total_bytes_alloc) != cudaSuccess ||
                cudaMalloc(&d_dst, total_bytes_alloc) != cudaSuccess) {
                if (d_src) cudaFree(d_src);
                cudaGetLastError();
                printf("%-22s %-18s %3d %6d %10s %10s %10s %8s %8s %8s\n",
                       shape_str.c_str(), perm_str.c_str(), wr, num_blocks,
                       "OOM", "-", "-", "-", "-", "-");
                continue;
            }
            cudaMemsetAsync(d_src, 0x3f, total_bytes_alloc, stream);
            cudaStreamSynchronize(stream);

            // Offsets: packed layout
            std::vector<int64_t> offsets(num_blocks);
            for (int b = 0; b < num_blocks; b++) offsets[b] = static_cast<int64_t>(b) * block_vol;

            // --- cuTENSOR: one plan, loop over blocks ---
            // Build permuted dims and strides
            uint32_t r = static_cast<uint32_t>(rank);
            std::vector<int64_t> src_s(r);
            src_s[r-1] = 1;
            for (int d = static_cast<int>(r)-2; d >= 0; --d)
                src_s[d] = src_s[d+1] * sc.dims[d+1];
            std::vector<int64_t> dst_d(r);
            for (uint32_t d = 0; d < r; d++) dst_d[d] = sc.dims[perm[d]];
            std::vector<int64_t> dst_s(r);
            dst_s[r-1] = 1;
            for (int d = static_cast<int>(r)-2; d >= 0; --d)
                dst_s[d] = dst_s[d+1] * dst_d[d+1];

            std::vector<int32_t> mA(r), mB(r);
            for (uint32_t d = 0; d < r; d++) mA[d] = static_cast<int32_t>(d);
            for (uint32_t d = 0; d < r; d++) mB[d] = perm[d];

            cudaDataType_t cdt = std::is_same<T,float>::value ? CUDA_R_32F : CUDA_R_64F;
            auto comp = std::is_same<T,float>::value
                ? CUTENSOR_COMPUTE_DESC_32F : CUTENSOR_COMPUTE_DESC_64F;
            const uint32_t align = static_cast<uint32_t>(sizeof(T));

            cutensorTensorDescriptor_t dA{}, dB{};
            cutensorCreateTensorDescriptor(handle, &dA, r, sc.dims.data(), src_s.data(), cdt, align);
            cutensorCreateTensorDescriptor(handle, &dB, r, dst_d.data(), dst_s.data(), cdt, align);

            cutensorOperationDescriptor_t op{};
            cutensorCreatePermutation(handle, &op, dA, mA.data(), CUTENSOR_OP_IDENTITY, dB, mB.data(), comp);
            cutensorPlanPreference_t pref{};
            cutensorCreatePlanPreference(handle, &pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE);
            uint64_t ws = 0;
            cutensorEstimateWorkspaceSize(handle, op, pref, CUTENSOR_WORKSPACE_DEFAULT, &ws);
            cutensorPlan_t ct_plan{};
            bool ct_ok = (cutensorCreatePlan(handle, &ct_plan, op, pref, ws) == CUTENSOR_STATUS_SUCCESS);
            T alpha = T(1);

            float ct_ms = 0;
            if (ct_ok) {
                int ct_bench_blocks = std::min(num_blocks, 1000);
                float ct_scale = static_cast<float>(num_blocks) / ct_bench_blocks;
                auto ct_result = gpu_bench([&]() {
                    for (int b = 0; b < ct_bench_blocks; b++)
                        cutensorPermute(handle, ct_plan, &alpha,
                                        d_src + offsets[b], d_dst + offsets[b], stream);
                }, warmup, repeat, stream);
                ct_ms = ct_result.median_ms * ct_scale;
            }
            if (ct_ok) cutensorDestroyPlan(ct_plan);
            cutensorDestroyPlanPreference(pref);
            cutensorDestroyOperationDescriptor(op);
            cutensorDestroyTensorDescriptor(dB);
            cutensorDestroyTensorDescriptor(dA);

            // --- Hypertile: single WR-grouped launch ---
            auto do_ht = [&](auto rank_tag) -> float {
                constexpr int R = decltype(rank_tag)::value;
                HypertileShape<R> hs = pack_shape<R>(task.shape);

                std::vector<HypertileMem> mems(num_blocks);
                for (int b = 0; b < num_blocks; b++) {
                    mems[b].src_base = offsets[b];
                    mems[b].dst_base = offsets[b];
                    mems[b].shape_idx = 0;
                }
                std::vector<int32_t> ps(num_blocks + 1);
                ps[0] = 0;
                for (int b = 0; b < num_blocks; b++)
                    ps[b+1] = ps[b] + static_cast<int32_t>(task.shape.num_tiles);

                void* d_shapes;
                void* d_mems;
                int32_t* d_prefix;
                cudaMalloc(&d_shapes, sizeof(HypertileShape<R>));
                cudaMalloc(&d_mems, num_blocks * sizeof(HypertileMem));
                cudaMalloc(&d_prefix, (num_blocks + 1) * sizeof(int32_t));
                cudaMemcpyAsync(d_shapes, &hs, sizeof(hs), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_mems, mems.data(), num_blocks * sizeof(HypertileMem),
                                cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_prefix, ps.data(), (num_blocks + 1) * sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream);
                cudaStreamSynchronize(stream);

                auto ht_result = gpu_bench([&]() {
                    launch_hypertile_kernel<T, 1024, 256>(
                        d_src, d_dst, d_shapes, d_mems, d_prefix,
                        num_blocks, ps[num_blocks], wr, stream);
                }, warmup, repeat, stream);

                cudaFree(d_shapes); cudaFree(d_mems); cudaFree(d_prefix);
                return ht_result.median_ms;
            };

            float ht_ms = 0;
            switch (wr) {
                case 2: ht_ms = do_ht(std::integral_constant<int,2>{}); break;
                case 3: ht_ms = do_ht(std::integral_constant<int,3>{}); break;
                case 4: ht_ms = do_ht(std::integral_constant<int,4>{}); break;
                case 5: ht_ms = do_ht(std::integral_constant<int,5>{}); break;
                case 6: ht_ms = do_ht(std::integral_constant<int,6>{}); break;
                case 7: ht_ms = do_ht(std::integral_constant<int,7>{}); break;
                case 8: ht_ms = do_ht(std::integral_constant<int,8>{}); break;
                default: break;
            }

            // Format volume
            char vol_str[32];
            if (total_vol >= 1000000)
                snprintf(vol_str, sizeof(vol_str), "%.1fM", total_vol / 1e6);
            else if (total_vol >= 1000)
                snprintf(vol_str, sizeof(vol_str), "%.1fK", total_vol / 1e3);
            else
                snprintf(vol_str, sizeof(vol_str), "%ld", (long)total_vol);

            double bytes = static_cast<double>(total_vol) * sizeof(T) * 2.0;
            double ht_gbps = ht_ms > 0 ? bytes / (ht_ms * 1e6) : 0;
            double ct_gbps = ct_ms > 0 ? bytes / (ct_ms * 1e6) : 0;
            float speedup = (ct_ms > 0 && ht_ms > 0) ? ct_ms / ht_ms : 0;

            if (ct_ms > 0 && ht_ms > 0) {
                printf("%-22s %-18s %3d %6d %10s %9.3fms %9.3fms %7.2fx %7.1f %7.1f\n",
                       shape_str.c_str(), perm_str.c_str(), wr, num_blocks, vol_str,
                       ct_ms, ht_ms, speedup, ht_gbps, ct_gbps);
            } else {
                printf("%-22s %-18s %3d %6d %10s %10s %10s %8s %8s %8s\n",
                       shape_str.c_str(), perm_str.c_str(), wr, num_blocks, vol_str,
                       ct_ok ? "ERROR" : "SKIP", ht_ms > 0 ? "OK" : "SKIP",
                       "-", "-", "-");
            }

            cudaFree(d_src);
            cudaFree(d_dst);
        }
    }
}


// ---------------------------------------------------------------------------
// EXP-1b(b): Grouped transpose with real DMRG permutations
// ---------------------------------------------------------------------------

namespace fs = std::filesystem;

// Helper: compute gather perm (cuTENSOR/benchmark convention) from scatter perm (pipeline convention)
static std::vector<int32_t> invert_perm(const std::vector<int32_t>& scatter) {
    std::vector<int32_t> gather(scatter.size());
    for (size_t i = 0; i < scatter.size(); ++i)
        gather[scatter[i]] = static_cast<int32_t>(i);
    return gather;
}

// One operand result for EXP-1b(b)
template <typename T>
static void bench_exp1b_b_operand(
    const std::string& case_name, const char* op_label,
    const std::vector<int32_t>& gather_perm,  // cuTENSOR convention: dst[i] = src[perm[i]]
    const BlockSparseTensor& tensor,
    int warmup, int repeat, cudaStream_t stream)
{
    int rank = static_cast<int>(tensor.meta.indices.size());
    auto perm_str = perm_to_string(gather_perm);

    // Extract per-block shapes and group by shape
    std::map<std::vector<int64_t>, std::vector<int>> shape_groups;
    std::vector<std::vector<int64_t>> block_shapes;
    std::vector<int64_t> block_vols;

    for (const auto& coord : tensor.block_coords) {
        std::vector<int64_t> shape;
        shape.reserve(rank);
        for (int d = 0; d < rank; d++) {
            char label = tensor.meta.indices[d];
            shape.push_back(tensor.block_sizes.at(label)[coord[d]]);
        }
        int blk = static_cast<int>(block_shapes.size());
        shape_groups[shape].push_back(blk);
        int64_t vol = 1;
        for (auto s : shape) vol *= s;
        block_vols.push_back(vol);
        block_shapes.push_back(std::move(shape));
    }

    int num_blocks = static_cast<int>(block_shapes.size());
    int num_shapes = static_cast<int>(shape_groups.size());

    std::vector<int64_t> offsets(num_blocks);
    int64_t total_vol = 0;
    for (int b = 0; b < num_blocks; b++) {
        offsets[b] = total_vol;
        total_vol += block_vols[b];
    }

    int64_t total_bytes = total_vol * static_cast<int64_t>(sizeof(T));
    if (total_bytes > 4LL * 1024 * 1024 * 1024) {
        printf("%-46s %2s %-18s %6d %6d %10s %10s %10s %8s %8s %8s\n",
               case_name.c_str(), op_label, perm_str.c_str(), num_shapes, num_blocks,
               "SKIP", "-", "-", "-", "-", "-");
        return;
    }

    T *d_src = nullptr, *d_dst = nullptr;
    if (cudaMalloc(&d_src, total_vol * sizeof(T)) != cudaSuccess ||
        cudaMalloc(&d_dst, total_vol * sizeof(T)) != cudaSuccess) {
        if (d_src) cudaFree(d_src);
        cudaGetLastError();
        printf("%-46s %2s %-18s %6d %6d %10s %10s %10s %8s %8s %8s\n",
               case_name.c_str(), op_label, perm_str.c_str(), num_shapes, num_blocks,
               "OOM", "-", "-", "-", "-", "-");
        return;
    }
    cudaMemsetAsync(d_src, 0x3f, total_vol * sizeof(T), stream);
    cudaStreamSynchronize(stream);

    // --- cuTENSOR ---
    cutensorHandle_t handle = get_handle();
    struct ShapePlanInfo { cutensorPlan_t plan{}; };
    std::map<std::vector<int64_t>, ShapePlanInfo> ct_plans;
    bool ct_ok = true;

    for (auto& [shape, blk_indices] : shape_groups) {
        uint32_t r = static_cast<uint32_t>(rank);
        std::vector<int64_t> src_s(r);
        src_s[r-1] = 1;
        for (int d = static_cast<int>(r)-2; d >= 0; --d)
            src_s[d] = src_s[d+1] * shape[d+1];
        std::vector<int64_t> dst_d(r);
        for (uint32_t d = 0; d < r; d++) dst_d[d] = shape[gather_perm[d]];
        std::vector<int64_t> dst_s(r);
        dst_s[r-1] = 1;
        for (int d = static_cast<int>(r)-2; d >= 0; --d)
            dst_s[d] = dst_s[d+1] * dst_d[d+1];

        std::vector<int32_t> mA(r), mB(r);
        for (uint32_t d = 0; d < r; d++) mA[d] = static_cast<int32_t>(d);
        for (uint32_t d = 0; d < r; d++) mB[d] = gather_perm[d];

        cudaDataType_t cdt = std::is_same<T,float>::value ? CUDA_R_32F : CUDA_R_64F;
        auto comp = std::is_same<T,float>::value
            ? CUTENSOR_COMPUTE_DESC_32F : CUTENSOR_COMPUTE_DESC_64F;
        const uint32_t al = static_cast<uint32_t>(sizeof(T));

        cutensorTensorDescriptor_t dA{}, dB{};
        auto s1 = cutensorCreateTensorDescriptor(handle, &dA, r, shape.data(), src_s.data(), cdt, al);
        auto s2 = cutensorCreateTensorDescriptor(handle, &dB, r, dst_d.data(), dst_s.data(), cdt, al);
        if (s1 != CUTENSOR_STATUS_SUCCESS || s2 != CUTENSOR_STATUS_SUCCESS) { ct_ok = false; break; }

        cutensorOperationDescriptor_t op{};
        cutensorCreatePermutation(handle, &op, dA, mA.data(), CUTENSOR_OP_IDENTITY, dB, mB.data(), comp);
        cutensorPlanPreference_t pref{};
        cutensorCreatePlanPreference(handle, &pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE);
        uint64_t ws = 0;
        cutensorEstimateWorkspaceSize(handle, op, pref, CUTENSOR_WORKSPACE_DEFAULT, &ws);

        ShapePlanInfo pi;
        if (cutensorCreatePlan(handle, &pi.plan, op, pref, ws) != CUTENSOR_STATUS_SUCCESS)
            { ct_ok = false; break; }
        ct_plans[shape] = pi;
        cutensorDestroyPlanPreference(pref);
        cutensorDestroyOperationDescriptor(op);
        cutensorDestroyTensorDescriptor(dB);
        cutensorDestroyTensorDescriptor(dA);
    }

    float ct_ms = 0;
    if (ct_ok) {
        std::vector<cutensorPlan_t> blk_plan(num_blocks);
        for (int b = 0; b < num_blocks; b++)
            blk_plan[b] = ct_plans[block_shapes[b]].plan;
        T alpha = T(1);
        int ct_bench_blocks = std::min(num_blocks, 1000);
        float ct_scale = static_cast<float>(num_blocks) / ct_bench_blocks;
        auto ct_result = gpu_bench([&]() {
            for (int b = 0; b < ct_bench_blocks; b++)
                cutensorPermute(handle, blk_plan[b], &alpha,
                                d_src + offsets[b], d_dst + offsets[b], stream);
        }, warmup, repeat, stream);
        ct_ms = ct_result.median_ms * ct_scale;
    }
    for (auto& [_, pi] : ct_plans) cutensorDestroyPlan(pi.plan);

    // --- Hypertile: WR-grouped ---
    std::vector<int32_t> inv_perm(rank);
    for (int i = 0; i < rank; i++) inv_perm[gather_perm[i]] = i;

    struct WRGroup {
        int wr = 0;
        std::vector<uint8_t> shape_bytes;
        int num_shapes = 0;
        std::vector<HypertileMem> mems;
        std::vector<uint32_t> tiles_per_task;
    };
    std::map<int, WRGroup> wr_groups;

    for (auto& sg : shape_groups) {
        auto& shape = sg.first;
        auto& blk_indices = sg.second;
        dim_t dd[16];
        for (int i = 0; i < rank; i++) dd[i] = static_cast<dim_t>(shape[i]);
        auto r = build_task_for_shape(dd, rank, rank, inv_perm.data(), gather_perm.data(), 1024);
        if (r.skip) continue;
        int wr = r.shape.WR;
        auto& grp = wr_groups[wr];
        grp.wr = wr;
        uint32_t shape_idx = static_cast<uint32_t>(grp.num_shapes);

        auto do_pack = [&](auto rank_tag) {
            constexpr int R = decltype(rank_tag)::value;
            HypertileShape<R> hs = pack_shape<R>(r.shape);
            auto* p = reinterpret_cast<const uint8_t*>(&hs);
            grp.shape_bytes.insert(grp.shape_bytes.end(), p, p + sizeof(hs));
        };
        switch (wr) {
            case 2: do_pack(std::integral_constant<int,2>{}); break;
            case 3: do_pack(std::integral_constant<int,3>{}); break;
            case 4: do_pack(std::integral_constant<int,4>{}); break;
            case 5: do_pack(std::integral_constant<int,5>{}); break;
            case 6: do_pack(std::integral_constant<int,6>{}); break;
            case 7: do_pack(std::integral_constant<int,7>{}); break;
            case 8: do_pack(std::integral_constant<int,8>{}); break;
            default: break;
        }
        grp.num_shapes++;
        for (int bi : blk_indices) {
            HypertileMem m{};
            m.src_base = offsets[bi];
            m.dst_base = offsets[bi];
            m.shape_idx = shape_idx;
            grp.mems.push_back(m);
            grp.tiles_per_task.push_back(r.shape.num_tiles);
        }
    }

    struct HTSlice {
        void* d_shapes = nullptr;
        void* d_mems = nullptr;
        int32_t* d_prefix = nullptr;
        int num_tasks = 0;
        int32_t total_tiles = 0;
        int wr = 0;
    };
    std::vector<HTSlice> ht_slices;
    for (auto& [wr, grp] : wr_groups) {
        if (grp.mems.empty()) continue;
        int nb = static_cast<int>(grp.mems.size());
        std::vector<int> order(nb);
        std::iota(order.begin(), order.end(), 0);
        auto& mems_ref = grp.mems;
        auto& tiles_ref = grp.tiles_per_task;
        std::sort(order.begin(), order.end(), [&mems_ref](int a, int b) {
            return mems_ref[a].shape_idx < mems_ref[b].shape_idx;
        });
        std::vector<HypertileMem> sorted_mems(nb);
        std::vector<int32_t> ps(nb + 1);
        ps[0] = 0;
        for (int i = 0; i < nb; i++) {
            sorted_mems[i] = mems_ref[order[i]];
            ps[i + 1] = ps[i] + static_cast<int32_t>(tiles_ref[order[i]]);
        }
        HTSlice sl{};
        sl.num_tasks = nb;
        sl.total_tiles = ps[nb];
        sl.wr = wr;
        cudaMalloc(&sl.d_shapes, grp.shape_bytes.size());
        cudaMalloc(&sl.d_mems, nb * sizeof(HypertileMem));
        cudaMalloc(&sl.d_prefix, (nb + 1) * sizeof(int32_t));
        cudaMemcpyAsync(sl.d_shapes, grp.shape_bytes.data(), grp.shape_bytes.size(), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(sl.d_mems, sorted_mems.data(), nb * sizeof(HypertileMem), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(sl.d_prefix, ps.data(), (nb + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        ht_slices.push_back(sl);
    }
    cudaStreamSynchronize(stream);

    float ht_ms = 0;
    if (!ht_slices.empty()) {
        auto ht_result = gpu_bench([&]() {
            for (auto& sl : ht_slices)
                launch_hypertile_kernel<T, 1024, 256>(
                    d_src, d_dst, sl.d_shapes, sl.d_mems, sl.d_prefix,
                    sl.num_tasks, sl.total_tiles, sl.wr, stream);
        }, warmup, repeat, stream);
        ht_ms = ht_result.median_ms;
    }
    for (auto& sl : ht_slices) { cudaFree(sl.d_shapes); cudaFree(sl.d_mems); cudaFree(sl.d_prefix); }

    // Format + print
    char vol_str[32];
    if (total_vol >= 1000000) snprintf(vol_str, sizeof(vol_str), "%.1fM", total_vol / 1e6);
    else if (total_vol >= 1000) snprintf(vol_str, sizeof(vol_str), "%.1fK", total_vol / 1e3);
    else snprintf(vol_str, sizeof(vol_str), "%ld", (long)total_vol);

    double bytes = static_cast<double>(total_vol) * sizeof(T) * 2.0;
    double ht_gbps = ht_ms > 0 ? bytes / (ht_ms * 1e6) : 0;
    double ct_gbps = ct_ms > 0 ? bytes / (ct_ms * 1e6) : 0;
    float speedup = (ct_ms > 0 && ht_ms > 0) ? ct_ms / ht_ms : 0;

    if (ct_ms > 0 && ht_ms > 0) {
        printf("%-46s %2s %-18s %6d %6d %10s %9.3fms %9.3fms %7.2fx %7.1f %7.1f\n",
               case_name.c_str(), op_label, perm_str.c_str(), num_shapes, num_blocks, vol_str,
               ct_ms, ht_ms, speedup, ht_gbps, ct_gbps);
    } else if (ht_ms > 0) {
        printf("%-46s %2s %-18s %6d %6d %10s %10s %9.3fms %8s %7.1f %8s\n",
               case_name.c_str(), op_label, perm_str.c_str(), num_shapes, num_blocks, vol_str,
               "ERROR", ht_ms, "-", ht_gbps, "-");
    } else {
        printf("%-46s %2s %-18s %6d %6d %10s %10s %10s %8s %8s %8s\n",
               case_name.c_str(), op_label, perm_str.c_str(), num_shapes, num_blocks, vol_str,
               "SKIP", "SKIP", "-", "-", "-");
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

template <typename T>
static void run_exp1b_b(const std::string& data_root, int warmup, int repeat, cudaStream_t stream) {
    printf("\n=== EXP-1b(b): Grouped Transpose — Real Permutations ===\n");
    printf("%-46s %2s %-18s %6s %6s %10s %10s %10s %8s %8s %8s\n",
           "Case", "Op", "Perm", "Shapes", "Blocks", "TotalVol",
           "cuTENSOR", "Hypertile", "Speedup", "HT GB/s", "CT GB/s");
    printf("%-46s %2s %-18s %6s %6s %10s %10s %10s %8s %8s %8s\n",
           std::string(46, '-').c_str(), "--", std::string(18, '-').c_str(),
           "------", "------", "--------", "--------", "---------",
           "-------", "-------", "-------");

    DataType dtype = std::is_same<T, double>::value ? DataType::FLOAT64 : DataType::FLOAT32;

    // Only canonical_formulation cases need physical transpose
    std::vector<std::string> case_dirs;
    for (const auto& entry : fs::directory_iterator(data_root)) {
        if (!entry.is_directory()) continue;
        std::string dirname = entry.path().filename().string();
        if (dirname.find("canonical_formulation") == std::string::npos) continue;
        if (dirname.find("_E") == std::string::npos &&
            dirname.find("_S") == std::string::npos) continue;
        case_dirs.push_back(dirname);
    }
    std::sort(case_dirs.begin(), case_dirs.end());

    for (const auto& dirname : case_dirs) {
        std::string dir_path = data_root + "/" + dirname;
        std::string fileA = dir_path + "/" + dirname + "_A.bs";
        std::string fileB = dir_path + "/" + dirname + "_B.bs";
        std::string fileEinsum = dir_path + "/" + dirname + "_einsum.txt";

        if (!fs::exists(fileA) || !fs::exists(fileB)) continue;

        BlockSparseTensor A, B;
        try {
            A = parse_bs_tensor_file(fileA, dtype, /*skip_values=*/true);
            B = parse_bs_tensor_file(fileB, dtype, /*skip_values=*/true);
        } catch (const std::exception& e) {
            printf("%-46s  SKIP (%s)\n", dirname.c_str(), e.what());
            continue;
        }

        // Read einsum from file or infer from metadata
        std::string einsum_str;
        if (fs::exists(fileEinsum)) {
            std::ifstream ifs(fileEinsum);
            std::getline(ifs, einsum_str);
        }
        if (einsum_str.empty()) {
            // Infer: A_labels,B_labels->output_labels
            // Output labels = labels in A or B but not both (free) + labels in both (batch)
            std::string a_str(A.meta.indices.begin(), A.meta.indices.end());
            std::string b_str(B.meta.indices.begin(), B.meta.indices.end());
            std::string out;
            // batch + free_A + free_B (order from output dir name or just concatenate)
            std::set<char> a_set(a_str.begin(), a_str.end());
            std::set<char> b_set(b_str.begin(), b_str.end());
            // Batch: in both
            for (char c : a_str) { if (b_set.count(c)) out += c; }
            // Free A: in A only
            for (char c : a_str) { if (!b_set.count(c)) out += c; }
            // Free B: in B only
            for (char c : b_str) { if (!a_set.count(c)) out += c; }
            einsum_str = a_str + "," + b_str + "->" + out;
        }

        // Parse einsum and build contraction config
        EinsumExpression expr;
        try {
            expr = parse_einsum(einsum_str);
        } catch (...) {
            printf("%-46s  SKIP (bad einsum: %s)\n", dirname.c_str(), einsum_str.c_str());
            continue;
        }
        EinsumIndexSets idx;
        try {
            create_einsum_index_sets(expr, idx);
        } catch (...) {
            printf("%-46s  SKIP (classify failed)\n", dirname.c_str());
            continue;
        }

        // Build contraction config inline (avoids pulling in pipeline.cuh)
        auto A_labels = std::vector<char>(A.meta.indices.begin(), A.meta.indices.end());
        auto B_labels = std::vector<char>(B.meta.indices.begin(), B.meta.indices.end());
        ContractionConfig config{};
        auto find_pos = [](const std::vector<char>& labels, char c) -> int32_t {
            for (size_t i = 0; i < labels.size(); ++i)
                if (labels[i] == c) return static_cast<int32_t>(i);
            return -1;
        };
        for (char c : idx.batch_indices) {
            int32_t pa = find_pos(A_labels, c), pb = find_pos(B_labels, c);
            if (pa >= 0 && pb >= 0) {
                config.batch_pos_A[config.num_batch] = pa;
                config.batch_pos_B[config.num_batch] = pb;
                config.num_batch++;
            }
        }
        for (char c : idx.free_indices_A) {
            int32_t pa = find_pos(A_labels, c);
            if (pa >= 0) config.free_pos_A[config.num_free_A++] = pa;
        }
        for (char c : idx.free_indices_B) {
            int32_t pb = find_pos(B_labels, c);
            if (pb >= 0) config.free_pos_B[config.num_free_B++] = pb;
        }
        for (char c : idx.contracted_indices) {
            int32_t pa = find_pos(A_labels, c), pb = find_pos(B_labels, c);
            if (pa >= 0 && pb >= 0) {
                config.contracted_pos_A[config.num_contracted] = pa;
                config.contracted_pos_B[config.num_contracted] = pb;
                config.num_contracted++;
            }
        }

        GemmLayoutPlan layout_plan = plan_gemm_layout(config,
            static_cast<dim_t>(A.meta.ndim), static_cast<dim_t>(B.meta.ndim));

        // Bench each operand that needs transpose
        if (layout_plan.transpose_A) {
            auto gather_perm = invert_perm(layout_plan.plan_A.perm);
            bench_exp1b_b_operand<T>(dirname, "A", gather_perm, A, warmup, repeat, stream);
        }
        if (layout_plan.transpose_B) {
            auto gather_perm = invert_perm(layout_plan.plan_B.perm);
            bench_exp1b_b_operand<T>(dirname, "B", gather_perm, B, warmup, repeat, stream);
        }
        if (!layout_plan.transpose_A && !layout_plan.transpose_B) {
            printf("%-46s  NO TRANSPOSE\n", dirname.c_str());
        }
    }
}


// ---------------------------------------------------------------------------
// Grouped transpose benchmark: real DMRG block shapes, random permutation
// ---------------------------------------------------------------------------

template <typename T>
static void run_grouped_bench(const std::string& data_root, uint32_t seed,
                              int warmup, int repeat, cudaStream_t stream) {
    printf("\n=== Grouped Transpose: Real DMRG Blocks ===\n");
    printf("%-50s %6s %6s %10s %10s %10s %8s %8s %8s\n",
           "Case", "Shapes", "Blocks", "TotalVol", "cuTENSOR", "Hypertile",
           "Speedup", "HT GB/s", "CT GB/s");
    printf("%-50s %6s %6s %10s %10s %10s %8s %8s %8s\n",
           std::string(50, '-').c_str(), "------", "------", "--------",
           "--------", "---------", "-------", "-------", "-------");

    DataType dtype = std::is_same<T, double>::value ? DataType::FLOAT64 : DataType::FLOAT32;
    cutensorHandle_t handle = get_handle();

    // Discover case directories
    std::vector<std::string> case_dirs;
    for (const auto& entry : fs::directory_iterator(data_root)) {
        if (!entry.is_directory()) continue;
        std::string dirname = entry.path().filename().string();
        if (dirname.find("_E") == std::string::npos &&
            dirname.find("_S") == std::string::npos) continue;
        case_dirs.push_back(dirname);
    }
    std::sort(case_dirs.begin(), case_dirs.end());

    for (const auto& dirname : case_dirs) {
        std::string dir_path = data_root + "/" + dirname;
        std::string fileA = dir_path + "/" + dirname + "_A.bs";

        if (!fs::exists(fileA)) continue;

        // Parse tensor A (metadata only, no values)
        BlockSparseTensor A;
        try {
            A = parse_bs_tensor_file(fileA, dtype, /*skip_values=*/true);
        } catch (const std::exception& e) {
            printf("%-50s  SKIP (%s)\n", dirname.c_str(), e.what());
            continue;
        }

        int rank = static_cast<int>(A.meta.indices.size());
        if (rank < 2 || rank > 8) {
            printf("%-50s  SKIP (rank=%d)\n", dirname.c_str(), rank);
            continue;
        }

        // Generate a deterministic non-identity permutation for this rank
        auto perm = make_random_perm(rank, seed);

        // Extract per-block shapes and group by shape
        std::map<std::vector<int64_t>, std::vector<int>> shape_groups;
        std::vector<std::vector<int64_t>> block_shapes;  // per-block
        std::vector<int64_t> block_vols;

        for (const auto& coord : A.block_coords) {
            std::vector<int64_t> shape;
            shape.reserve(rank);
            for (int d = 0; d < rank; d++) {
                char label = A.meta.indices[d];
                shape.push_back(A.block_sizes.at(label)[coord[d]]);
            }
            int blk = static_cast<int>(block_shapes.size());
            shape_groups[shape].push_back(blk);
            int64_t vol = 1;
            for (auto s : shape) vol *= s;
            block_vols.push_back(vol);
            block_shapes.push_back(std::move(shape));
        }

        int num_blocks = static_cast<int>(block_shapes.size());
        int num_shapes = static_cast<int>(shape_groups.size());

        // Compute offsets (packed layout)
        std::vector<int64_t> offsets(num_blocks);
        int64_t total_vol = 0;
        for (int b = 0; b < num_blocks; b++) {
            offsets[b] = total_vol;
            total_vol += block_vols[b];
        }

        int64_t total_bytes = total_vol * static_cast<int64_t>(sizeof(T));
        if (total_bytes > 4LL * 1024 * 1024 * 1024) {
            printf("%-50s %6d %6d %10s %10s %10s %8s %8s %8s\n",
                   dirname.c_str(), num_shapes, num_blocks, "SKIP", "-", "-", "-", "-", "-");
            continue;
        }

        // Allocate GPU buffers
        T *d_src = nullptr, *d_dst = nullptr;
        if (cudaMalloc(&d_src, total_vol * sizeof(T)) != cudaSuccess ||
            cudaMalloc(&d_dst, total_vol * sizeof(T)) != cudaSuccess) {
            if (d_src) cudaFree(d_src);
            cudaGetLastError();
            printf("%-50s %6d %6d %10s %10s %10s %8s %8s %8s\n",
                   dirname.c_str(), num_shapes, num_blocks, "OOM", "-", "-", "-", "-", "-");
            continue;
        }
        cudaMemsetAsync(d_src, 0x3f, total_vol * sizeof(T), stream);
        cudaStreamSynchronize(stream);

        // --- cuTENSOR: one plan per unique shape, loop over blocks ---
        struct ShapePlanInfo {
            cutensorPlan_t plan{};
            T alpha = T(1);
        };
        std::map<std::vector<int64_t>, ShapePlanInfo> ct_plans;

        bool ct_ok = true;
        for (auto& [shape, blk_indices] : shape_groups) {
            uint32_t r = static_cast<uint32_t>(rank);
            std::vector<int64_t> src_s(r);
            src_s[r-1] = 1;
            for (int d = static_cast<int>(r)-2; d >= 0; --d)
                src_s[d] = src_s[d+1] * shape[d+1];
            std::vector<int64_t> dst_d(r);
            for (uint32_t d = 0; d < r; d++) dst_d[d] = shape[perm[d]];
            std::vector<int64_t> dst_s(r);
            dst_s[r-1] = 1;
            for (int d = static_cast<int>(r)-2; d >= 0; --d)
                dst_s[d] = dst_s[d+1] * dst_d[d+1];

            std::vector<int32_t> mA(r), mB(r);
            for (uint32_t d = 0; d < r; d++) mA[d] = static_cast<int32_t>(d);
            for (uint32_t d = 0; d < r; d++) mB[d] = perm[d];

            cudaDataType_t cdt = std::is_same<T,float>::value ? CUDA_R_32F : CUDA_R_64F;
            auto comp = std::is_same<T,float>::value
                ? CUTENSOR_COMPUTE_DESC_32F : CUTENSOR_COMPUTE_DESC_64F;

            const uint32_t align = static_cast<uint32_t>(sizeof(T));
            cutensorTensorDescriptor_t dA{}, dB{};
            auto s1 = cutensorCreateTensorDescriptor(handle, &dA, r,
                shape.data(), src_s.data(), cdt, align);
            auto s2 = cutensorCreateTensorDescriptor(handle, &dB, r,
                dst_d.data(), dst_s.data(), cdt, align);
            if (s1 != CUTENSOR_STATUS_SUCCESS || s2 != CUTENSOR_STATUS_SUCCESS) {
                ct_ok = false; break;
            }

            cutensorOperationDescriptor_t op{};
            cutensorCreatePermutation(handle, &op, dA, mA.data(), CUTENSOR_OP_IDENTITY,
                dB, mB.data(), comp);
            cutensorPlanPreference_t pref{};
            cutensorCreatePlanPreference(handle, &pref,
                CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE);
            uint64_t ws = 0;
            cutensorEstimateWorkspaceSize(handle, op, pref,
                CUTENSOR_WORKSPACE_DEFAULT, &ws);

            ShapePlanInfo pi;
            auto sp = cutensorCreatePlan(handle, &pi.plan, op, pref, ws);
            if (sp != CUTENSOR_STATUS_SUCCESS) { ct_ok = false; break; }

            ct_plans[shape] = pi;
            cutensorDestroyPlanPreference(pref);
            cutensorDestroyOperationDescriptor(op);
            cutensorDestroyTensorDescriptor(dB);
            cutensorDestroyTensorDescriptor(dA);
        }

        float ct_ms = 0;
        if (ct_ok) {
            // Build block→plan mapping for fast dispatch
            std::vector<cutensorPlan_t> blk_plan(num_blocks);
            for (int b = 0; b < num_blocks; b++)
                blk_plan[b] = ct_plans[block_shapes[b]].plan;
            T alpha = T(1);

            int ct_bench_blocks = std::min(num_blocks, 1000);
            float ct_scale = static_cast<float>(num_blocks) / ct_bench_blocks;

            auto ct_result = gpu_bench([&]() {
                for (int b = 0; b < ct_bench_blocks; b++) {
                    cutensorPermute(handle, blk_plan[b], &alpha,
                                    d_src + offsets[b], d_dst + offsets[b], stream);
                }
            }, warmup, repeat, stream);
            ct_ms = ct_result.median_ms * ct_scale;
        }

        for (auto& [_, pi] : ct_plans) cutensorDestroyPlan(pi.plan);

        // --- Hypertile: group all blocks by WR, one kernel per WR ---
        struct HTSlice {
            void* d_shapes = nullptr;
            void* d_mems = nullptr;
            int32_t* d_prefix = nullptr;
            int num_tasks = 0;
            int32_t total_tiles = 0;
            int wr = 0;
        };

        // Per-WR accumulators (host side)
        struct WRGroup {
            int wr = 0;
            std::vector<uint8_t> shape_bytes;   // packed HypertileShape<R>[]
            int num_shapes = 0;
            std::vector<HypertileMem> mems;
            std::vector<uint32_t> tiles_per_task; // num_tiles for each mem entry
        };
        std::map<int, WRGroup> wr_groups;  // keyed by WR

        std::vector<int32_t> inv_perm(rank);
        for (int i = 0; i < rank; i++) inv_perm[perm[i]] = i;

        for (auto& sg : shape_groups) {
            auto& shape = sg.first;
            auto& blk_indices = sg.second;
            dim_t dd[16];
            for (int i = 0; i < rank; i++) dd[i] = static_cast<dim_t>(shape[i]);

            auto r = build_task_for_shape(dd, rank, rank, inv_perm.data(), perm.data(), 1024);
            if (r.skip) continue;
            int wr = r.shape.WR;

            auto& grp = wr_groups[wr];
            grp.wr = wr;
            uint32_t shape_idx = static_cast<uint32_t>(grp.num_shapes);

            // Pack shape into byte buffer (type-erased, dispatch by WR)
            auto do_pack = [&](auto rank_tag) {
                constexpr int R = decltype(rank_tag)::value;
                HypertileShape<R> hs = pack_shape<R>(r.shape);
                auto* p = reinterpret_cast<const uint8_t*>(&hs);
                grp.shape_bytes.insert(grp.shape_bytes.end(), p, p + sizeof(hs));
            };
            switch (wr) {
                case 2: do_pack(std::integral_constant<int,2>{}); break;
                case 3: do_pack(std::integral_constant<int,3>{}); break;
                case 4: do_pack(std::integral_constant<int,4>{}); break;
                case 5: do_pack(std::integral_constant<int,5>{}); break;
                case 6: do_pack(std::integral_constant<int,6>{}); break;
                case 7: do_pack(std::integral_constant<int,7>{}); break;
                case 8: do_pack(std::integral_constant<int,8>{}); break;
                default: break;
            }
            grp.num_shapes++;

            // Add mems for all blocks of this shape, sorted by shape_idx
            for (int bi : blk_indices) {
                HypertileMem m{};
                m.src_base = offsets[bi];
                m.dst_base = offsets[bi];
                m.shape_idx = shape_idx;
                grp.mems.push_back(m);
                grp.tiles_per_task.push_back(r.shape.num_tiles);
            }
        }

        // Sort mems by shape_idx within each WR group (for kernel cache reuse)
        // and build prefix sums, then upload
        std::vector<HTSlice> ht_slices;
        for (auto& [wr, grp] : wr_groups) {
            if (grp.mems.empty()) continue;
            int nb = static_cast<int>(grp.mems.size());

            // Sort by shape_idx
            auto& mems_ref = grp.mems;
            auto& tiles_ref = grp.tiles_per_task;
            std::vector<int> order(nb);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&mems_ref](int a, int b) {
                return mems_ref[a].shape_idx < mems_ref[b].shape_idx;
            });
            std::vector<HypertileMem> sorted_mems(nb);
            std::vector<int32_t> ps(nb + 1);
            ps[0] = 0;
            for (int i = 0; i < nb; i++) {
                sorted_mems[i] = mems_ref[order[i]];
                ps[i + 1] = ps[i] + static_cast<int32_t>(tiles_ref[order[i]]);
            }

            HTSlice sl{};
            sl.num_tasks = nb;
            sl.total_tiles = ps[nb];
            sl.wr = wr;
            cudaMalloc(&sl.d_shapes, grp.shape_bytes.size());
            cudaMalloc(&sl.d_mems, nb * sizeof(HypertileMem));
            cudaMalloc(&sl.d_prefix, (nb + 1) * sizeof(int32_t));
            cudaMemcpyAsync(sl.d_shapes, grp.shape_bytes.data(),
                            grp.shape_bytes.size(), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(sl.d_mems, sorted_mems.data(),
                            nb * sizeof(HypertileMem), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(sl.d_prefix, ps.data(),
                            (nb + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
            ht_slices.push_back(sl);
        }
        cudaStreamSynchronize(stream);

        float ht_ms = 0;
        if (!ht_slices.empty()) {
            auto ht_result = gpu_bench([&]() {
                for (auto& sl : ht_slices) {
                    launch_hypertile_kernel<T, 1024, 256>(
                        d_src, d_dst, sl.d_shapes, sl.d_mems, sl.d_prefix,
                        sl.num_tasks, sl.total_tiles, sl.wr, stream);
                }
            }, warmup, repeat, stream);
            ht_ms = ht_result.median_ms;
        }

        for (auto& sl : ht_slices) {
            cudaFree(sl.d_shapes);
            cudaFree(sl.d_mems);
            cudaFree(sl.d_prefix);
        }

        // Format total volume
        char vol_str[32];
        if (total_vol >= 1000000)
            snprintf(vol_str, sizeof(vol_str), "%.1fM", total_vol / 1e6);
        else if (total_vol >= 1000)
            snprintf(vol_str, sizeof(vol_str), "%.1fK", total_vol / 1e3);
        else
            snprintf(vol_str, sizeof(vol_str), "%ld", (long)total_vol);

        double bytes = static_cast<double>(total_vol) * sizeof(T) * 2.0;
        double ht_gbps = ht_ms > 0 ? bytes / (ht_ms * 1e6) : 0;
        double ct_gbps = ct_ms > 0 ? bytes / (ct_ms * 1e6) : 0;
        float speedup = (ct_ms > 0 && ht_ms > 0) ? ct_ms / ht_ms : 0;

        if (ct_ms > 0 && ht_ms > 0) {
            printf("%-50s %6d %6d %10s %9.3fms %9.3fms %7.2fx %7.1f %7.1f\n",
                   dirname.c_str(), num_shapes, num_blocks, vol_str,
                   ct_ms, ht_ms, speedup, ht_gbps, ct_gbps);
        } else if (ht_ms > 0) {
            printf("%-50s %6d %6d %10s %10s %9.3fms %8s %7.1f %8s\n",
                   dirname.c_str(), num_shapes, num_blocks, vol_str,
                   "ERROR", ht_ms, "-", ht_gbps, "-");
        } else {
            printf("%-50s %6d %6d %10s %10s %10s %8s %8s %8s\n",
                   dirname.c_str(), num_shapes, num_blocks, vol_str,
                   "SKIP", "SKIP", "-", "-", "-");
        }

        cudaFree(d_src);
        cudaFree(d_dst);
    }
}


int main(int argc, char** argv) {
    int warmup = 3;
    int repeat = 5;
    bool use_f64 = false;
    uint32_t seed = 42;
    std::string shape_str, perm_str;
    int64_t total_vol = 0;  // 0 = use default

    std::string grouped_dir;
    bool run_exp1b_a_flag = false;
    std::string exp1b_b_dir;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (arg == "--repeat" && i + 1 < argc) repeat = std::atoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (arg == "--shape" && i + 1 < argc) shape_str = argv[++i];
        else if (arg == "--perm" && i + 1 < argc) perm_str = argv[++i];
        else if (arg == "--total-vol" && i + 1 < argc) total_vol = std::atol(argv[++i]);
        else if (arg == "--grouped" && i + 1 < argc) grouped_dir = argv[++i];
        else if (arg == "--exp1b-a") run_exp1b_a_flag = true;
        else if (arg == "--exp1b-b" && i + 1 < argc) exp1b_b_dir = argv[++i];
        else if (arg == "--dtype" && i + 1 < argc) {
            std::string d = argv[++i];
            use_f64 = (d == "f64" || d == "double");
        }
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (%.1f GB, SM %d.%d)\n",
           prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);

    // Single-case reproduce mode
    if (!shape_str.empty() && !perm_str.empty()) {
        auto dims = parse_shape(shape_str);
        auto perm = parse_perm(perm_str);
        if (use_f64) run_single_case<double>(dims, perm, warmup, repeat, stream);
        else         run_single_case<float>(dims, perm, warmup, repeat, stream);
        cudaStreamDestroy(stream);
        return 0;
    }

    printf("Settings: warmup=%d, repeat=%d, dtype=%s, seed=%u\n\n",
           warmup, repeat, use_f64 ? "f64" : "f32", seed);

    bool run_mixed = false;
    bool run_batch_sweep = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--mixed") run_mixed = true;
        if (std::string(argv[i]) == "--batch-sweep") run_batch_sweep = true;
    }

    // EXP-1b modes
    if (run_exp1b_a_flag) {
        if (use_f64) run_exp1b_a<double>(warmup, repeat, stream);
        else         run_exp1b_a<float>(warmup, repeat, stream);
    }
    if (!exp1b_b_dir.empty()) {
        if (use_f64) run_exp1b_b<double>(exp1b_b_dir, warmup, repeat, stream);
        else         run_exp1b_b<float>(exp1b_b_dir, warmup, repeat, stream);
    }
    if (run_exp1b_a_flag || !exp1b_b_dir.empty()) {
        // Also run grouped if requested alongside exp1b
        if (!grouped_dir.empty()) {
            if (use_f64) run_grouped_bench<double>(grouped_dir, seed, warmup, repeat, stream);
            else         run_grouped_bench<float>(grouped_dir, seed, warmup, repeat, stream);
        }
        cudaStreamDestroy(stream);
        return 0;
    }

    if (!grouped_dir.empty()) {
        if (use_f64) run_grouped_bench<double>(grouped_dir, seed, warmup, repeat, stream);
        else         run_grouped_bench<float>(grouped_dir, seed, warmup, repeat, stream);
        cudaStreamDestroy(stream);
        return 0;
    }

    if (run_batch_sweep) {
        // ~1 GB default: 256M f32 elements or 128M f64 elements
        int64_t default_vol = use_f64 ? (128L << 20) : (256L << 20);
        int64_t vol = total_vol > 0 ? total_vol : default_vol;
        if (use_f64) run_batch_sweep_bench<double>(warmup, repeat, stream, seed, vol);
        else         run_batch_sweep_bench<float>(warmup, repeat, stream, seed, vol);
    } else {
        if (use_f64) {
            run_single_bench<double>(warmup, repeat, stream, seed);
            run_batched_bench<double>(warmup, repeat, stream, seed);
            if (run_mixed) run_mixed_bench<double>(warmup, repeat, stream);
        } else {
            run_single_bench<float>(warmup, repeat, stream, seed);
            run_batched_bench<float>(warmup, repeat, stream, seed);
            if (run_mixed) run_mixed_bench<float>(warmup, repeat, stream);
        }
    }

    cudaStreamDestroy(stream);
    return 0;
}
