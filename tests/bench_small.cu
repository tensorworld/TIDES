// Per-kernel benchmark for warp-level small GEMM kernel.
// Builds WarpShape + WarpMem arrays directly and times launch_warp_gemm
// across representative configs: standard NN/TN, interleaved large dim,
// interleaved K, and interleaved large+K.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "general_sparse/pipeline.cuh"

using namespace bsptc;

// ---------------------------------------------------------------------------
// Timing harness
// ---------------------------------------------------------------------------

struct BenchResult { float median_ms, min_ms, max_ms; };

template <typename Fn>
BenchResult gpu_bench(Fn fn, int warmup, int repeat, cudaStream_t stream) {
    for (int i = 0; i < warmup; i++) { fn(); }
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
// Helper: upload shape + mems + warp prefix to GPU
// ---------------------------------------------------------------------------

template <typename ShapeT>
struct GpuBuffers {
    void* d_shapes = nullptr;
    WarpMem* d_mems = nullptr;
    int32_t* d_wp = nullptr;
    int num_descs = 0;
    int total_warps = 0;

    void upload(const ShapeT& shape, const std::vector<WarpMem>& h_mems,
                const std::vector<int32_t>& h_warp_prefix) {
        num_descs = static_cast<int>(h_mems.size());
        total_warps = h_warp_prefix.back();
        cudaMalloc(&d_shapes, sizeof(ShapeT));
        cudaMalloc(&d_mems, num_descs * sizeof(WarpMem));
        cudaMalloc(&d_wp, (num_descs + 1) * sizeof(int32_t));
        cudaMemcpy(d_shapes, &shape, sizeof(ShapeT), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mems, h_mems.data(), num_descs * sizeof(WarpMem), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wp, h_warp_prefix.data(), (num_descs + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    void free() {
        cudaFree(d_shapes);
        cudaFree(d_mems);
        cudaFree(d_wp);
    }
};

// ---------------------------------------------------------------------------
// Helper: build a ShapeSS for standard (non-interleaved) GEMM
// ---------------------------------------------------------------------------

static ShapeSS make_standard_shape(int M, int N, int K, bool a_k_inner) {
    ShapeSS s{};
    bool m_small = (M <= N);
    s.small_size = m_small ? M : N;
    s.large_size = m_small ? N : M;
    s.K_total = K;
    s.m_is_small = m_small ? 1 : 0;

    dim_t stride_a_free, stride_b_free;
    if (a_k_inner) {  // NN: A=[M,K], B=[K,N]
        stride_a_free = K;   // M stride in A
        stride_b_free = 1;   // N stride in B
        s.k.stride_a = 1;
        s.k.stride_b = N;
    } else {           // TN: A=[K,M], B=[K,N]
        stride_a_free = 1;   // M stride in A
        stride_b_free = 1;   // N stride in B
        s.k.stride_a = M;
        s.k.stride_b = N;
    }
    s.small_stride = m_small ? stride_a_free : stride_b_free;
    s.large.stride = m_small ? stride_b_free : stride_a_free;
    return s;
}

// ---------------------------------------------------------------------------
// Standard GEMM benchmark (DescSS): NN and TN modes
// ---------------------------------------------------------------------------

template <typename T>
static void bench_standard(int warmup, int repeat, cudaStream_t stream) {
    printf("\n=== Warp GEMM: Standard (DescSS) ===\n");
    printf("%-6s %5s %5s %5s %-4s %7s %10s %8s\n",
           "NDesc", "M", "N", "K", "Mode", "Warps", "Median(ms)", "GFLOP/s");
    printf("%-6s %5s %5s %5s %-4s %7s %10s %8s\n",
           "------", "-----", "-----", "-----", "----", "-------", "----------", "--------");

    int num_descs_arr[] = {100, 1000, 5000, 20000};
    struct MNK { int M, N, K; };
    MNK shapes[] = {
        {4, 4, 4}, {8, 8, 8}, {16, 16, 16}, {32, 32, 32},
        {4, 4, 64}, {1, 1, 256}, {64, 1, 32}, {8, 8, 128}
    };

    struct ModeConfig {
        const char* name;
        bool a_k_inner;        // true = NN (K is innermost in A)
    };
    // variant = (INT_LARGE<<3)|(INT_K<<2)|(CKA<<1)|CKB
    // INT_LARGE=0, INT_K=0
    // NN: CKA=1, CKB=0 → variant=2;  TN: CKA=0, CKB=0 → variant=0
    ModeConfig modes[] = {
        {"NN",  true},    // variant=2
        {"TN",  false},   // variant=0
    };

    for (auto& mode : modes) {
        int variant = mode.a_k_inner ? 2 : 0;
        for (auto np : num_descs_arr) {
            for (auto shape : shapes) {
                int M = shape.M, N = shape.N, K = shape.K;
                int64_t total_elems = (int64_t)np * (M * K + K * N + M * N);

                T* d_buf = nullptr;
                cudaMalloc(&d_buf, total_elems * sizeof(T));
                cudaMemset(d_buf, 0, total_elems * sizeof(T));

                ShapeSS s = make_standard_shape(M, N, K, mode.a_k_inner);

                int32_t warps_per = (M * N + 31) / 32;
                std::vector<WarpMem> h_mems(np);
                std::vector<int32_t> h_warp_prefix(np + 1);
                h_warp_prefix[0] = 0;
                int64_t off = 0;
                for (int i = 0; i < np; i++) {
                    h_mems[i].A = d_buf + off; off += M * K;
                    h_mems[i].B = d_buf + off; off += K * N;
                    h_mems[i].C = d_buf + off; off += M * N;
                    h_mems[i].shape_idx = 0;
                    h_mems[i].use_atomic = 0;
                    h_warp_prefix[i + 1] = h_warp_prefix[i] + warps_per;
                }

                GpuBuffers<ShapeSS> gb;
                gb.upload(s, h_mems, h_warp_prefix);

                auto result = gpu_bench([&]() {
                    launch_warp_gemm<T>(gb.d_shapes, gb.d_mems, gb.d_wp,
                                        gb.num_descs, gb.total_warps,
                                        variant, stream);
                }, warmup, repeat, stream);

                double gflops = 2.0 * np * M * N * K / (result.median_ms * 1e6);
                printf("%6d %5d %5d %5d %-4s %7d %10.3f %8.1f\n",
                       np, M, N, K, mode.name, gb.total_warps, result.median_ms, gflops);

                gb.free();
                cudaFree(d_buf);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Interleaved large dim benchmark (DescIS): large dim decomposed into 2 dims
// ---------------------------------------------------------------------------

template <typename T>
static void bench_interleaved_large(int warmup, int repeat, cudaStream_t stream) {
    printf("\n=== Warp GEMM: Interleaved Large (DescIS) ===\n");
    printf("%-6s %8s %5s %5s %7s %10s %8s\n",
           "NDesc", "Lg(d0xd1)", "Sm", "K", "Warps", "Median(ms)", "GFLOP/s");
    printf("%-6s %8s %5s %5s %7s %10s %8s\n",
           "------", "---------", "-----", "-----", "-------", "----------", "--------");

    // Large dim = M = d0 * d1 (interleaved free dims in A), Small dim = N
    struct Config { int np; int d0, d1; int N, K; };
    Config configs[] = {
        {1000,  4,  4, 4,  16},   // M=16 (large), N=4 (small)
        {1000,  8,  4, 8,  32},   // M=32, N=8
        {5000,  4,  2, 4,   8},   // M=8, N=4
        {5000,  4,  4, 4,  64},   // M=16, N=4
        {20000, 2,  2, 2,   4},   // M=4, N=2
    };

    // variant = (INT_LARGE<<3)|(INT_K<<2)|(CKA<<1)|CKB
    // INT_LARGE=1, INT_K=0, NN: CKA=1, CKB=0 → variant = 8|0|2|0 = 10
    int variant = 10;

    using ShapeIS = WarpShape<IntDim, StdK>;

    for (auto& cfg : configs) {
        int M = cfg.d0 * cfg.d1, N = cfg.N, K = cfg.K;
        int64_t total_elems = (int64_t)cfg.np * (M * K + K * N + M * N);

        T* d_buf = nullptr;
        cudaMalloc(&d_buf, total_elems * sizeof(T));
        cudaMemset(d_buf, 0, total_elems * sizeof(T));

        // A=[d0, K, d1] layout (interleaved free dims in A)
        // A strides: d0 → K*d1, k → d1, d1 → 1
        int A_stride_d0 = K * cfg.d1;
        int A_stride_d1 = 1;

        ShapeIS s{};
        // N < M in all configs → N is small
        s.small_size = N;
        s.large_size = M;
        s.K_total = K;
        s.m_is_small = 0;  // N is small (M is large)
        s.small_stride = 1; // N stride in B (B=[K,N], N innermost)

        // Large dim is M = A's free (interleaved)
        s.large.num_dims = 2;
        s.large.dims[0] = cfg.d0;
        s.large.dims[1] = cfg.d1;
        s.large.strides[0] = A_stride_d0;
        s.large.strides[1] = A_stride_d1;

        s.k.stride_a = cfg.d1;  // K stride in A (middle dim)
        s.k.stride_b = N;       // K stride in B

        int32_t warps_per = (M * N + 31) / 32;
        std::vector<WarpMem> h_mems(cfg.np);
        std::vector<int32_t> h_warp_prefix(cfg.np + 1);
        h_warp_prefix[0] = 0;
        int64_t off = 0;
        for (int i = 0; i < cfg.np; i++) {
            h_mems[i].A = d_buf + off; off += M * K;
            h_mems[i].B = d_buf + off; off += K * N;
            h_mems[i].C = d_buf + off; off += M * N;
            h_mems[i].shape_idx = 0;
            h_mems[i].use_atomic = 0;
            h_warp_prefix[i + 1] = h_warp_prefix[i] + warps_per;
        }

        GpuBuffers<ShapeIS> gb;
        gb.upload(s, h_mems, h_warp_prefix);

        auto result = gpu_bench([&]() {
            launch_warp_gemm<T>(gb.d_shapes, gb.d_mems, gb.d_wp,
                                gb.num_descs, gb.total_warps, variant, stream);
        }, warmup, repeat, stream);

        double gflops = 2.0 * cfg.np * M * N * K / (result.median_ms * 1e6);
        printf("%6d %4dx%-3d %5d %5d %7d %10.3f %8.1f\n",
               cfg.np, cfg.d0, cfg.d1, N, K, gb.total_warps, result.median_ms, gflops);

        gb.free();
        cudaFree(d_buf);
    }
}

// ---------------------------------------------------------------------------
// Interleaved K benchmark (DescSI): K decomposed into 2 dims
// ---------------------------------------------------------------------------

template <typename T>
static void bench_interleaved_k(int warmup, int repeat, cudaStream_t stream) {
    printf("\n=== Warp GEMM: Interleaved K (DescSI) ===\n");
    printf("%-6s %5s %5s %8s %7s %10s %8s\n",
           "NDesc", "M", "N", "K(d0xd1)", "Warps", "Median(ms)", "GFLOP/s");
    printf("%-6s %5s %5s %8s %7s %10s %8s\n",
           "------", "-----", "-----", "--------", "-------", "----------", "--------");

    struct Config { int np; int M, N; int kd0, kd1; };
    Config configs[] = {
        {1000,  4, 4,  4,  4},    // K=16
        {1000,  8, 8,  4,  8},    // K=32
        {5000,  4, 4,  2,  8},    // K=16
        {5000,  4, 4,  4, 16},    // K=64
        {20000, 2, 2,  2,  4},    // K=8
    };

    // variant = (INT_LARGE<<3)|(INT_K<<2)|(CKA<<1)|CKB
    // INT_LARGE=0, INT_K=1, CKA=1, CKB=1 → variant = 0|4|2|1 = 7
    int variant = 7;

    using ShapeSI = WarpShape<StdDim, IntK>;

    for (auto& cfg : configs) {
        int K = cfg.kd0 * cfg.kd1;
        int64_t total_elems = (int64_t)cfg.np * (cfg.M * K + K * cfg.N + cfg.M * cfg.N);

        T* d_buf = nullptr;
        cudaMalloc(&d_buf, total_elems * sizeof(T));
        cudaMemset(d_buf, 0, total_elems * sizeof(T));

        ShapeSI s{};
        bool m_small = (cfg.M <= cfg.N);
        s.small_size = m_small ? cfg.M : cfg.N;
        s.large_size = m_small ? cfg.N : cfg.M;
        s.K_total = K;
        s.m_is_small = m_small ? 1 : 0;
        // NN: A=[M,K], B=[K,N]. M stride in A = K, N stride in B = 1
        s.small_stride = m_small ? K : 1;
        s.large.stride = m_small ? 1 : K;

        s.k.num_dims = 2;
        s.k.dims[0] = cfg.kd0;
        s.k.dims[1] = cfg.kd1;
        s.k.strides_a[0] = cfg.kd1;
        s.k.strides_a[1] = 1;
        s.k.strides_b[0] = cfg.kd1 * cfg.N;
        s.k.strides_b[1] = cfg.N;
        s.k.k_start = 0;

        int32_t warps_per = (cfg.M * cfg.N + 31) / 32;
        std::vector<WarpMem> h_mems(cfg.np);
        std::vector<int32_t> h_warp_prefix(cfg.np + 1);
        h_warp_prefix[0] = 0;
        int64_t off = 0;
        for (int i = 0; i < cfg.np; i++) {
            h_mems[i].A = d_buf + off; off += cfg.M * K;
            h_mems[i].B = d_buf + off; off += K * cfg.N;
            h_mems[i].C = d_buf + off; off += cfg.M * cfg.N;
            h_mems[i].shape_idx = 0;
            h_mems[i].use_atomic = 0;
            h_warp_prefix[i + 1] = h_warp_prefix[i] + warps_per;
        }

        GpuBuffers<ShapeSI> gb;
        gb.upload(s, h_mems, h_warp_prefix);

        auto result = gpu_bench([&]() {
            launch_warp_gemm<T>(gb.d_shapes, gb.d_mems, gb.d_wp,
                                gb.num_descs, gb.total_warps, variant, stream);
        }, warmup, repeat, stream);

        double gflops = 2.0 * cfg.np * cfg.M * cfg.N * K / (result.median_ms * 1e6);
        printf("%6d %5d %5d %4dx%-3d %7d %10.3f %8.1f\n",
               cfg.np, cfg.M, cfg.N, cfg.kd0, cfg.kd1, gb.total_warps, result.median_ms, gflops);

        gb.free();
        cudaFree(d_buf);
    }
}

// ---------------------------------------------------------------------------
// Interleaved large+K benchmark (DescII): large dim and K both decomposed
// ---------------------------------------------------------------------------

template <typename T>
static void bench_interleaved_large_k(int warmup, int repeat, cudaStream_t stream) {
    printf("\n=== Warp GEMM: Interleaved Large+K (DescII) ===\n");
    printf("%-6s %8s %5s %8s %7s %10s %8s\n",
           "NDesc", "Lg(d0xd1)", "Sm", "K(d0xd1)", "Warps", "Median(ms)", "GFLOP/s");
    printf("%-6s %8s %5s %8s %7s %10s %8s\n",
           "------", "---------", "-----", "--------", "-------", "----------", "--------");

    // A=[f0, k0, f1, k1] (interleaved), B=[k0, k1, g]
    // M = f0*f1 (large, interleaved in A), N = g (small), K = k0*k1 (interleaved)
    struct Config { int np; int f0, f1, g, kd0, kd1; };
    Config configs[] = {
        {1000,  4, 2, 2, 2, 4},    // M=8 (large), N=2 (small), K=8
        {1000,  4, 2, 4, 2, 8},    // M=8, N=4, K=16
        {5000,  2, 2, 2, 2, 2},    // M=4, N=2, K=4
        {5000,  4, 4, 2, 2, 4},    // M=16, N=2, K=8
        {20000, 2, 2, 1, 2, 2},    // M=4, N=1, K=4
    };

    // variant = (INT_LARGE<<3)|(INT_K<<2)|(CKA<<1)|CKB
    // INT_LARGE=1, INT_K=1, CKA=1, CKB=1 → variant = 8|4|2|1 = 15
    int variant = 15;

    using ShapeII = WarpShape<IntDim, IntK>;

    for (auto& cfg : configs) {
        int M = cfg.f0 * cfg.f1;
        int N = cfg.g;
        int K = cfg.kd0 * cfg.kd1;
        int64_t total_elems = (int64_t)cfg.np * (M * K + K * N + M * N);

        T* d_buf = nullptr;
        cudaMalloc(&d_buf, total_elems * sizeof(T));
        cudaMemset(d_buf, 0, total_elems * sizeof(T));

        // A=[f0, k0, f1, k1]: strides
        int A_stride_f0 = cfg.kd0 * cfg.f1 * cfg.kd1;
        int A_stride_k0 = cfg.f1 * cfg.kd1;
        int A_stride_f1 = cfg.kd1;
        int A_stride_k1 = 1;
        // B=[k0, k1, g]: strides
        int B_stride_k0 = cfg.kd1 * cfg.g;
        int B_stride_k1 = cfg.g;
        int B_stride_g  = 1;

        ShapeII s{};
        // N < M in all configs → N is small
        s.small_size = N;
        s.large_size = M;
        s.K_total = K;
        s.m_is_small = 0;       // N is small
        s.small_stride = B_stride_g;  // N stride in B

        // Large dim is M = A's free (interleaved: f0, f1)
        s.large.num_dims = 2;
        s.large.dims[0] = cfg.f0;
        s.large.dims[1] = cfg.f1;
        s.large.strides[0] = A_stride_f0;
        s.large.strides[1] = A_stride_f1;

        // K (interleaved: k0, k1)
        s.k.num_dims = 2;
        s.k.dims[0] = cfg.kd0;
        s.k.dims[1] = cfg.kd1;
        s.k.strides_a[0] = A_stride_k0;
        s.k.strides_a[1] = A_stride_k1;
        s.k.strides_b[0] = B_stride_k0;
        s.k.strides_b[1] = B_stride_k1;
        s.k.k_start = 0;

        int32_t warps_per = (M * N + 31) / 32;
        std::vector<WarpMem> h_mems(cfg.np);
        std::vector<int32_t> h_warp_prefix(cfg.np + 1);
        h_warp_prefix[0] = 0;
        int64_t off = 0;
        for (int i = 0; i < cfg.np; i++) {
            h_mems[i].A = d_buf + off; off += M * K;
            h_mems[i].B = d_buf + off; off += K * N;
            h_mems[i].C = d_buf + off; off += M * N;
            h_mems[i].shape_idx = 0;
            h_mems[i].use_atomic = 0;
            h_warp_prefix[i + 1] = h_warp_prefix[i] + warps_per;
        }

        GpuBuffers<ShapeII> gb;
        gb.upload(s, h_mems, h_warp_prefix);

        auto result = gpu_bench([&]() {
            launch_warp_gemm<T>(gb.d_shapes, gb.d_mems, gb.d_wp,
                                gb.num_descs, gb.total_warps, variant, stream);
        }, warmup, repeat, stream);

        double gflops = 2.0 * cfg.np * M * N * K / (result.median_ms * 1e6);
        char lg_str[16], k_str[16];
        snprintf(lg_str, sizeof(lg_str), "%dx%d", cfg.f0, cfg.f1);
        snprintf(k_str, sizeof(k_str), "%dx%d", cfg.kd0, cfg.kd1);
        printf("%6d %9s %5d %8s %7d %10.3f %8.1f\n",
               cfg.np, lg_str, N, k_str, gb.total_warps, result.median_ms, gflops);

        gb.free();
        cudaFree(d_buf);
    }
}

// ---------------------------------------------------------------------------
// Batch benchmark (DescSS with batch_size > 1)
// ---------------------------------------------------------------------------

template <typename T>
static void bench_batched(int warmup, int repeat, cudaStream_t stream) {
    printf("\n=== Warp GEMM: Batched (DescSS, batch_size>1) ===\n");
    printf("%-6s %5s %5s %5s %5s %7s %10s %8s\n",
           "NDesc", "M", "N", "K", "Batch", "Warps", "Median(ms)", "GFLOP/s");
    printf("%-6s %5s %5s %5s %5s %7s %10s %8s\n",
           "------", "-----", "-----", "-----", "-----", "-------", "----------", "--------");

    struct Config { int np; int M, N, K, batch; };
    Config configs[] = {
        {100,  4, 4, 4, 8},
        {1000, 4, 4, 8, 4},
        {1000, 8, 8, 8, 8},
        {5000, 4, 4, 4, 4},
    };

    // NN, INT_LARGE=0, INT_K=0, CKA=1, CKB=0 → variant=2
    int variant = 2;

    for (auto& cfg : configs) {
        int64_t per_batch_A = cfg.M * cfg.K;
        int64_t per_batch_B = cfg.K * cfg.N;
        int64_t per_batch_C = cfg.M * cfg.N;
        int64_t total_elems = (int64_t)cfg.np * cfg.batch * (per_batch_A + per_batch_B + per_batch_C);

        T* d_buf = nullptr;
        cudaMalloc(&d_buf, total_elems * sizeof(T));
        cudaMemset(d_buf, 0, total_elems * sizeof(T));

        ShapeSS s = make_standard_shape(cfg.M, cfg.N, cfg.K, /*a_k_inner=*/true);

        // Expand batches on host: each batch slice becomes a separate desc
        int num_descs = cfg.np * cfg.batch;
        int32_t warps_per = (cfg.M * cfg.N + 31) / 32;
        std::vector<WarpMem> h_mems(num_descs);
        std::vector<int32_t> h_warp_prefix(num_descs + 1);
        h_warp_prefix[0] = 0;
        int64_t off = 0;
        for (int i = 0; i < cfg.np; i++) {
            T* base_A = d_buf + off; off += cfg.batch * per_batch_A;
            T* base_B = d_buf + off; off += cfg.batch * per_batch_B;
            T* base_C = d_buf + off; off += cfg.batch * per_batch_C;
            for (int b = 0; b < cfg.batch; b++) {
                int idx = i * cfg.batch + b;
                h_mems[idx].A = base_A + b * per_batch_A;
                h_mems[idx].B = base_B + b * per_batch_B;
                h_mems[idx].C = base_C + b * per_batch_C;
                h_mems[idx].shape_idx = 0;
                h_mems[idx].use_atomic = 0;
                h_warp_prefix[idx + 1] = h_warp_prefix[idx] + warps_per;
            }
        }

        GpuBuffers<ShapeSS> gb;
        gb.upload(s, h_mems, h_warp_prefix);

        auto result = gpu_bench([&]() {
            launch_warp_gemm<T>(gb.d_shapes, gb.d_mems, gb.d_wp,
                                gb.num_descs, gb.total_warps, variant, stream);
        }, warmup, repeat, stream);

        double gflops = 2.0 * cfg.np * cfg.batch * cfg.M * cfg.N * cfg.K / (result.median_ms * 1e6);
        printf("%6d %5d %5d %5d %5d %7d %10.3f %8.1f\n",
               cfg.np, cfg.M, cfg.N, cfg.K, cfg.batch, gb.total_warps, result.median_ms, gflops);

        gb.free();
        cudaFree(d_buf);
    }
}

// ---------------------------------------------------------------------------
// Einsum comparison: interleaved kernel vs transpose+standard using pipeline APIs.
// Builds synthetic GPUSparseInput tensors, uses plan_gemm_layout,
// build_hypertile_index, prepare_small_shapes + bind_small_mems from the pipeline.
// ---------------------------------------------------------------------------

static int rng_next(uint32_t& state, int lo, int hi) {
    state = state * 1103515245 + 12345;
    return lo + ((state >> 16) & 0x7fff) % (hi - lo + 1);
}

// Build synthetic GPUSparseInput with nb blocks of given dims (all same rank).
// block_dims_flat: [nb * rank] flat array of per-block dims.
// Values are allocated on GPU as a single zeroed buffer; each block gets a
// contiguous region. Returns gpu input (host metadata) and d_values pointer.
template <typename T>
static GPUSparseInput<T> build_synthetic_gpu_input(
    int nb, dim_t rank,
    const std::vector<dim_t>& block_dims_flat,
    const std::vector<char>& labels)
{
    GPUSparseInput<T> gpu;
    gpu.rank = rank;
    gpu.num_blocks = static_cast<dim_t>(nb);
    gpu.labels = labels;
    gpu.block_dims = block_dims_flat;
    gpu.block_offsets.resize(nb);
    gpu.block_volumes.resize(nb);
    // Dummy coords (not needed for transpose/desc building)
    gpu.block_coords.resize(nb * rank, 0);

    int64_t offset = 0;
    for (int i = 0; i < nb; i++) {
        int64_t vol = 1;
        for (dim_t d = 0; d < rank; d++)
            vol *= block_dims_flat[i * rank + d];
        gpu.block_offsets[i] = offset;
        gpu.block_volumes[i] = vol;
        offset += vol;
    }
    gpu.total_values = offset;
    return gpu;
}

// Allocate GPU values and build a minimal DeviceSparseInput (only d_values used).
template <typename T>
static DeviceSparseInput<T> upload_synthetic_values(GPUSparseInput<T>& gpu) {
    DeviceSparseInput<T> dev{};
    dev.rank = gpu.rank;
    dev.num_blocks = gpu.num_blocks;
    dev.total_elements = gpu.total_values;
    check_cuda(cudaMalloc(&dev.d_values, dev.total_elements * sizeof(T)), "malloc synth values");
    check_cuda(cudaMemset(dev.d_values, 0, dev.total_elements * sizeof(T)), "memset synth values");
    // Upload block_dims (needed for transpose precomp which reads gpu.block_dims on host)
    // Also upload block_offsets (needed by transpose desc builder)
    return dev;
}

// Run one einsum config: build interleaved descs and transpose+standard descs,
// time both, return {interleaved_ms, standard_ms} (kernel only, no transpose)
// and transpose_ms separately.
struct EinsumBenchResult {
    float interleaved_ms;
    float transpose_ms;
    float standard_ms;
};

template <typename T>
static EinsumBenchResult bench_one_einsum(
    const char* einsum_str,
    const std::vector<dim_t>& A_block_dims_flat,  // [nb * A_rank]
    const std::vector<dim_t>& B_block_dims_flat,  // [nb * B_rank]
    int nb,
    int warmup, int repeat, cudaStream_t stream)
{
    // Parse einsum to get config
    auto expr = parse_einsum(einsum_str);
    EinsumIndexSets isets;
    create_einsum_index_sets(expr, isets);
    std::vector<char> A_labels(expr.inputs[0].labels.begin(), expr.inputs[0].labels.end());
    std::vector<char> B_labels(expr.inputs[1].labels.begin(), expr.inputs[1].labels.end());
    dim_t A_rank = static_cast<dim_t>(A_labels.size());
    dim_t B_rank = static_cast<dim_t>(B_labels.size());
    auto config = build_contraction_config(isets, A_labels, B_labels, expr.output_labels);

    // Build synthetic GPU inputs
    auto A_gpu = build_synthetic_gpu_input<T>(nb, A_rank, A_block_dims_flat, A_labels);
    auto B_gpu = build_synthetic_gpu_input<T>(nb, B_rank, B_block_dims_flat, B_labels);
    auto A_dev = upload_synthetic_values(A_gpu);
    auto B_dev = upload_synthetic_values(B_gpu);

    // Build a minimal PipelinePlan for prepare_small_shapes + bind_small_mems
    // We need: config, layout_plan, gemm_mode, physical_transpose_needed,
    //          work.small.problems, accum_map, d_output, stream_small
    auto layout_plan = plan_gemm_layout(config, A_rank, B_rank);

    // Build GemmProblemDesc for each block (treat each as a small problem pair)
    // M = product of free dims in A, N = product of free dims in B, K = product of contracted
    std::vector<GemmProblemDesc> small_problems(nb);
    for (int i = 0; i < nb; i++) {
        const dim_t* Ad = &A_block_dims_flat[i * A_rank];
        const dim_t* Bd = &B_block_dims_flat[i * B_rank];
        dim_t M = 1, N = 1, K = 1, batch = 1;
        for (int32_t j = 0; j < config.num_free_A; j++) M *= Ad[config.free_pos_A[j]];
        for (int32_t j = 0; j < config.num_free_B; j++) N *= Bd[config.free_pos_B[j]];
        for (int32_t j = 0; j < config.num_contracted; j++) K *= Ad[config.contracted_pos_A[j]];
        for (int32_t j = 0; j < config.num_batch; j++) batch *= Ad[config.batch_pos_A[j]];

        auto& p = small_problems[i];
        p.pair_idx = i;
        p.block_idx_A = i;
        p.block_idx_B = i;
        p.M = M; p.N = N; p.K = K;
        p.batch_size = batch;
        p.lda = K; p.ldb = N; p.ldc = N;
        p.offset_A = A_gpu.block_offsets[i];
        p.offset_B = B_gpu.block_offsets[i];
        p.offset_C = 0;  // filled via accum_map below
    }

    // Build a simple accumulation map: each pair is its own unique output block.
    // Output offset for pair i = sum of M*N*batch for pairs 0..i-1.
    int64_t total_output = 0;
    std::vector<int64_t> dest_offsets(nb);
    for (int i = 0; i < nb; i++) {
        dest_offsets[i] = total_output;
        total_output += static_cast<int64_t>(small_problems[i].M)
                      * small_problems[i].N * small_problems[i].batch_size;
    }

    // Allocate output buffer
    T* d_output = nullptr;
    check_cuda(cudaMalloc(&d_output, total_output * sizeof(T)), "malloc output");
    check_cuda(cudaMemset(d_output, 0, total_output * sizeof(T)), "memset output");

    // Allocate raw dest_offsets array (AccumulationMap uses int64_t*, not vector)
    int64_t* h_dest_raw = new int64_t[nb];
    for (int i = 0; i < nb; i++) h_dest_raw[i] = dest_offsets[i];

    // === INTERLEAVED PATH ===
    PipelinePlan<T> plan_int{};
    plan_int.config = config;
    plan_int.layout_plan = layout_plan;
    plan_int.needs_transpose = (layout_plan.transpose_A || layout_plan.transpose_B) && !plan_int.work.large.empty();
    plan_int.gemm_mode = layout_plan.mode_after_transpose;
    plan_int.d_output = d_output;
    plan_int.work.small.problems = small_problems;
    plan_int.accum_map.h_dest_offsets = h_dest_raw;
    cudaStreamCreate(&plan_int.stream_small);

    prepare_small_shapes(plan_int, A_gpu, B_gpu);
    bind_small_mems(plan_int, A_dev, B_dev);

    auto res_int = gpu_bench([&]() {
        launch_warp_gemm<T>(plan_int.d_small_shapes, plan_int.d_small_mems,
            plan_int.d_warp_prefix, plan_int.small_num_descs,
            plan_int.total_warps, plan_int.warp_variant, plan_int.stream_small);
    }, warmup, repeat, plan_int.stream_small);

    // === TRANSPOSE + STANDARD PATH ===
    // Build transpose precomp
    TransposeFlatMeta precomp_A, precomp_B;
    if (layout_plan.plan_A.transpose) {
        auto smap_A = build_block_shape_map(A_gpu.block_dims.data(), A_gpu.num_blocks, A_gpu.rank);
        auto idx_A = build_hypertile_index<T>(A_gpu, layout_plan.plan_A, smap_A);
        auto triv_A = compute_trivial_shapes(smap_A, &idx_A);
        std::vector<int> all_A(A_gpu.num_blocks);
        std::iota(all_A.begin(), all_A.end(), 0);
        precomp_A = assemble_hypertile_batch(all_A, A_gpu.block_offsets.data(),
            A_gpu.block_volumes.data(), smap_A, idx_A, triv_A, {}, nullptr, stream);
    }
    if (layout_plan.plan_B.transpose) {
        auto smap_B = build_block_shape_map(B_gpu.block_dims.data(), B_gpu.num_blocks, B_gpu.rank);
        auto idx_B = build_hypertile_index<T>(B_gpu, layout_plan.plan_B, smap_B);
        auto triv_B = compute_trivial_shapes(smap_B, &idx_B);
        std::vector<int> all_B(B_gpu.num_blocks);
        std::iota(all_B.begin(), all_B.end(), 0);
        precomp_B = assemble_hypertile_batch(all_B, B_gpu.block_offsets.data(),
            B_gpu.block_volumes.data(), smap_B, idx_B, triv_B, {}, nullptr, stream);
    }
    cudaDeviceSynchronize();

    // Pre-allocate transpose workspace outside timing loop
    T* d_ws_A = nullptr;
    T* d_ws_B = nullptr;
    if (layout_plan.plan_A.transpose && precomp_A.valid)
        check_cuda(cudaMalloc(&d_ws_A, A_dev.total_elements * sizeof(T)), "malloc ws_A");
    if (layout_plan.plan_B.transpose && precomp_B.valid)
        check_cuda(cudaMalloc(&d_ws_B, B_dev.total_elements * sizeof(T)), "malloc ws_B");

    // Time transpose kernel only (workspace pre-allocated)
    auto res_transpose = gpu_bench([&]() {
        if (d_ws_A)
            launch_batched_transpose<T>(A_dev.d_values, d_ws_A, precomp_A, stream);
        if (d_ws_B)
            launch_batched_transpose<T>(B_dev.d_values, d_ws_B, precomp_B, stream);
    }, warmup, repeat, stream);

    // Run actual transpose for the standard kernel using pre-allocated workspace
    if (d_ws_A)
        launch_batched_transpose<T>(A_dev.d_values, d_ws_A, precomp_A, stream);
    if (d_ws_B)
        launch_batched_transpose<T>(B_dev.d_values, d_ws_B, precomp_B, stream);
    cudaDeviceSynchronize();

    GemmMode mode_after = layout_plan.mode_after_transpose;

    // Build standard path plan using transposed data — temporarily override d_values
    T* orig_A_vals = A_dev.d_values;
    T* orig_B_vals = B_dev.d_values;
    if (d_ws_A) A_dev.d_values = d_ws_A;
    if (d_ws_B) B_dev.d_values = d_ws_B;

    // Reset output
    check_cuda(cudaMemset(d_output, 0, total_output * sizeof(T)), "memset output2");

    PipelinePlan<T> plan_std{};
    plan_std.config = config;
    plan_std.layout_plan = layout_plan;
    plan_std.needs_transpose = false;  // data is already transposed
    plan_std.gemm_mode = mode_after;
    plan_std.d_output = d_output;
    plan_std.work.small.problems = small_problems;
    plan_std.accum_map.h_dest_offsets = h_dest_raw;
    cudaStreamCreate(&plan_std.stream_small);

    prepare_small_shapes(plan_std, A_gpu, B_gpu);
    bind_small_mems(plan_std, A_dev, B_dev);

    // Restore original d_values so DeviceSparseInput destructors free the right memory
    A_dev.d_values = orig_A_vals;
    B_dev.d_values = orig_B_vals;

    auto res_std = gpu_bench([&]() {
        launch_warp_gemm<T>(plan_std.d_small_shapes, plan_std.d_small_mems,
            plan_std.d_warp_prefix, plan_std.small_num_descs,
            plan_std.total_warps, plan_std.warp_variant, plan_std.stream_small);
    }, warmup, repeat, plan_std.stream_small);

    // Null out borrowed pointers so PipelinePlan destructors don't double-free
    plan_int.accum_map.h_dest_offsets = nullptr;
    plan_std.accum_map.h_dest_offsets = nullptr;
    plan_int.d_output = nullptr;
    plan_std.d_output = nullptr;
    // PipelinePlan destructors handle the rest (d_small_shapes/mems, streams, etc.)
    precomp_A.free_gpu();
    precomp_B.free_gpu();
    if (d_ws_A) cudaFree(d_ws_A);
    if (d_ws_B) cudaFree(d_ws_B);
    cudaFree(d_output);
    delete[] h_dest_raw;

    return {res_int.median_ms, res_transpose.median_ms, res_std.median_ms};
}

template <typename T>
static void bench_einsum_comparison(int warmup, int repeat, cudaStream_t stream) {
    printf("\n");
    printf("Einsum comparison: interleaved kernel vs transpose+standard kernel\n");
    printf("Uses pipeline APIs (prepare/bind_small)\n");
    printf("Standard path includes actual GPU transpose of the same interleaved data.\n");
    printf("Ratio = interleaved / (transpose + standard). >1 means interleaved slower.\n");

    // Dimension ranges. Small tier constraint: min(M, N) < 8.
    const int F0_LO = 2, F0_HI = 12;
    const int K_LO  = 8, K_HI  = 64;
    const int G_LO  = 1, G_HI  = 7;
    const int K0_LO = 2, K0_HI = 8;
    const int K1_LO = 2, K1_HI = 8;

    int inner_dims[] = {1, 2, 4, 8, 16, 32};
    const int MAX_WARPS = 2000000;

    auto compute_nb = [](int avg_mn, int max_warps, int min_nb = 500, int max_nb = 50000) {
        int avg_w = std::max(1, (avg_mn + 31) / 32);
        return std::max(min_nb, std::min(max_nb, max_warps / avg_w));
    };

    // ================================================================
    // INT_M: A=[f0, k, f1], B=[k, g]  (einsum: "aeb,ec->abc")
    // ================================================================
    printf("\n=== INT_M: A=[f0, k, f1]  B=[k, g]  ===\n\n");
    printf("%-6s %6s %6s %6s %6s %10s %10s %10s %8s\n",
           "f1", "NBlk", "AvgM", "AvgN", "AvgK", "Intlv(ms)", "Xpose(ms)", "Std(ms)", "Ratio");
    printf("%-6s %6s %6s %6s %6s %10s %10s %10s %8s\n",
           "------", "------", "------", "------", "------",
           "----------", "----------", "----------", "--------");

    for (int f1 : inner_dims) {
        int avg_mn = ((F0_LO + F0_HI) / 2) * f1 * ((G_LO + G_HI) / 2);
        int nb = compute_nb(avg_mn, MAX_WARPS);

        uint32_t seed = 12345;
        // A=[f0, k, f1] (rank 3), B=[k, g] (rank 2)
        std::vector<dim_t> A_dims(nb * 3), B_dims(nb * 2);
        int64_t sM = 0, sN = 0, sK = 0;
        for (int i = 0; i < nb; i++) {
            int v_f0 = rng_next(seed, F0_LO, F0_HI);
            int v_k  = rng_next(seed, K_LO, K_HI);
            int v_g  = rng_next(seed, G_LO, G_HI);
            A_dims[i * 3 + 0] = v_f0;
            A_dims[i * 3 + 1] = v_k;
            A_dims[i * 3 + 2] = f1;
            B_dims[i * 2 + 0] = v_k;
            B_dims[i * 2 + 1] = v_g;
            sM += v_f0 * f1; sN += v_g; sK += v_k;
        }

        auto r = bench_one_einsum<T>("aeb,ec->abc", A_dims, B_dims, nb,
                                  warmup, repeat, stream);
        float total_std = r.transpose_ms + r.standard_ms;
        printf("%6d %6d %6d %6d %6d %10.3f %10.3f %10.3f %7.2fx\n",
               f1, nb, (int)(sM / nb), (int)(sN / nb), (int)(sK / nb),
               r.interleaved_ms, r.transpose_ms, r.standard_ms,
               r.interleaved_ms / total_std);
    }

    // ================================================================
    // INT_MK: A=[f0, k0, f1, k1], B=[k0, k1, g]  (einsum: "adbc,dce->abe")
    // ================================================================
    printf("\n=== INT_MK: A=[f0, k0, f1, k1]  B=[k0, k1, g]  ===\n\n");
    printf("%-6s %6s %6s %6s %6s %10s %10s %10s %8s\n",
           "f1", "NBlk", "AvgM", "AvgN", "AvgK", "Intlv(ms)", "Xpose(ms)", "Std(ms)", "Ratio");
    printf("%-6s %6s %6s %6s %6s %10s %10s %10s %8s\n",
           "------", "------", "------", "------", "------",
           "----------", "----------", "----------", "--------");

    for (int f1 : inner_dims) {
        int avg_mn = ((F0_LO + F0_HI) / 2) * f1 * ((G_LO + G_HI) / 2);
        int nb = compute_nb(avg_mn, MAX_WARPS);

        uint32_t seed = 34567;
        // A=[f0, k0, f1, k1] (rank 4), B=[k0, k1, g] (rank 3)
        std::vector<dim_t> A_dims(nb * 4), B_dims(nb * 3);
        int64_t sM = 0, sN = 0, sK = 0;
        for (int i = 0; i < nb; i++) {
            int v_f0 = rng_next(seed, F0_LO, F0_HI);
            int v_k0 = rng_next(seed, K0_LO, K0_HI);
            int v_k1 = rng_next(seed, K1_LO, K1_HI);
            int v_g  = rng_next(seed, G_LO, G_HI);
            A_dims[i * 4 + 0] = v_f0;
            A_dims[i * 4 + 1] = v_k0;
            A_dims[i * 4 + 2] = f1;
            A_dims[i * 4 + 3] = v_k1;
            B_dims[i * 3 + 0] = v_k0;
            B_dims[i * 3 + 1] = v_k1;
            B_dims[i * 3 + 2] = v_g;
            sM += v_f0 * f1; sN += v_g; sK += v_k0 * v_k1;
        }

        auto r = bench_one_einsum<T>("adbc,dce->abe", A_dims, B_dims, nb,
                                  warmup, repeat, stream);
        float total_std = r.transpose_ms + r.standard_ms;
        printf("%6d %6d %6d %6d %6d %10.3f %10.3f %10.3f %7.2fx\n",
               f1, nb, (int)(sM / nb), (int)(sN / nb), (int)(sK / nb),
               r.interleaved_ms, r.transpose_ms, r.standard_ms,
               r.interleaved_ms / total_std);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

template <typename T>
static void run_all_benches(int warmup, int repeat, const std::string& filter, cudaStream_t stream) {
    if (filter.empty() || filter.find("std") != std::string::npos ||
        filter.find("standard") != std::string::npos) {
        bench_standard<T>(warmup, repeat, stream);
    }
    if (filter.empty() || filter.find("intl") != std::string::npos ||
        filter.find("interleaved_large") != std::string::npos ||
        filter.find("intm") != std::string::npos ||
        filter.find("interleaved_m") != std::string::npos) {
        bench_interleaved_large<T>(warmup, repeat, stream);
    }
    if (filter.empty() || filter.find("intk") != std::string::npos ||
        filter.find("interleaved_k") != std::string::npos) {
        bench_interleaved_k<T>(warmup, repeat, stream);
    }
    if (filter.empty() || filter.find("full") != std::string::npos ||
        filter.find("interleaved") != std::string::npos) {
        bench_interleaved_large_k<T>(warmup, repeat, stream);
    }
    if (filter.empty() || filter.find("batch") != std::string::npos) {
        bench_batched<T>(warmup, repeat, stream);
    }
    if (filter.empty() || filter.find("einsum") != std::string::npos ||
        filter.find("comparison") != std::string::npos) {
        bench_einsum_comparison<T>(warmup, repeat, stream);
    }
}

int main(int argc, char** argv) {
    int warmup = 3, repeat = 5;
    std::string filter;
    bool use_f64 = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--filter") && i + 1 < argc) filter = argv[++i];
        else if (!strcmp(argv[i], "--dtype") && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "f64" || v == "float64") use_f64 = true;
        }
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    if (use_f64)
        run_all_benches<double>(warmup, repeat, filter, stream);
    else
        run_all_benches<float>(warmup, repeat, filter, stream);

    cudaStreamDestroy(stream);
    return 0;
}
