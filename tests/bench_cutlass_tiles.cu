// Per-tile CUTLASS grouped GEMM benchmark.
// Measures raw GFLOP/s for each TileShape to calibrate efficiency coefficients
// in best_tile_for_queue (work_partitioner.hpp).
//
// Usage:
//   ./bench_cutlass_tiles                              # 1024x1024x128, sweep batch 1..1024
//   ./bench_cutlass_tiles --shape 512x512x256          # custom shape
//   ./bench_cutlass_tiles --num-problems 64             # fixed batch count
//   ./bench_cutlass_tiles --sweep                       # M/N sweep across all tiles
//   ./bench_cutlass_tiles --count-sweep                 # batch-count sweep on multiple shapes
//   ./bench_cutlass_tiles --dtype f64                   # double precision

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "general_sparse/kernels/large/cutlass_grouped_fwd.hpp"
#include "general_sparse/kernels/large/cutlass_tile_shapes.hpp"
#include "general_sparse/pipeline_descs.hpp"
#include "general_sparse/gpu_sparse_device.cuh"

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
// Build synthetic LargeBlockWork + device buffers for a uniform problem set
// ---------------------------------------------------------------------------

template <typename T>
struct SyntheticSetup {
    LargeBlockWork<T> work;
    T* d_A = nullptr;
    T* d_B = nullptr;
    T* d_C = nullptr;
    int64_t total_flops = 0;

    void free() {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
};

template <typename T>
SyntheticSetup<T> build_synthetic(int M, int N, int K, int num_problems) {
    SyntheticSetup<T> s;
    s.work.gemm_mode = GemmMode::NN;

    int64_t A_elems = static_cast<int64_t>(M) * K;
    int64_t B_elems = static_cast<int64_t>(K) * N;
    int64_t C_elems = static_cast<int64_t>(M) * N;
    int64_t total_A = A_elems * num_problems;
    int64_t total_B = B_elems * num_problems;
    int64_t total_C = C_elems * num_problems;

    cudaMalloc(&s.d_A, total_A * sizeof(T));
    cudaMalloc(&s.d_B, total_B * sizeof(T));
    cudaMalloc(&s.d_C, total_C * sizeof(T));

    // Zero-init (values don't matter for timing)
    cudaMemset(s.d_A, 0, total_A * sizeof(T));
    cudaMemset(s.d_B, 0, total_B * sizeof(T));
    cudaMemset(s.d_C, 0, total_C * sizeof(T));

    s.work.ptr_A.resize(num_problems);
    s.work.ptr_B.resize(num_problems);
    s.work.ptr_C.resize(num_problems);
    s.work.lda.resize(num_problems);
    s.work.ldb.resize(num_problems);
    s.work.ldc.resize(num_problems);
    s.work.ldd.resize(num_problems);
    s.work.num_gemm_problems = num_problems;
    s.work.total_output_elements = total_C;

    for (int i = 0; i < num_problems; ++i) {
        GemmProblemDesc desc{};
        desc.pair_idx = i;
        desc.M = M;
        desc.N = N;
        desc.K = K;
        desc.batch_size = 1;
        desc.lda = K;  // NN layout: A is MxK row-major
        desc.ldb = N;  // B is KxN row-major
        desc.ldc = N;
        s.work.problems.push_back(desc);

        s.work.ptr_A[i] = s.d_A + i * A_elems;
        s.work.ptr_B[i] = s.d_B + i * B_elems;
        s.work.ptr_C[i] = s.d_C + i * C_elems;
        s.work.lda[i] = K;
        s.work.ldb[i] = N;
        s.work.ldc[i] = N;
        s.work.ldd[i] = N;
    }

    s.total_flops = 2LL * M * N * K * num_problems;
    return s;
}

// ---------------------------------------------------------------------------
// Benchmark one tile shape at one (M, N, K) config
// ---------------------------------------------------------------------------

struct TileResult {
    TileShape tile;
    int M, N, K, num_problems;
    float median_ms;
    double gflops;
};

template <typename T>
TileResult bench_tile(TileShape tile, int M, int N, int K,
                      int num_problems, int warmup, int repeat,
                      cudaStream_t stream)
{
    auto setup = build_synthetic<T>(M, N, K, num_problems);

    // Prepare CUTLASS grouped GEMM
    T alpha = T(1), beta = T(0);
    auto runner = prepare_cutlass_gemm_dispatch<T>(
        setup.work, alpha, beta, tile, stream);

    if (!runner) {
        setup.free();
        return {tile, M, N, K, num_problems, -1.0f, 0.0};
    }

    auto result = gpu_bench([&]() {
        runner.run(stream);
    }, warmup, repeat, stream);

    double gflops = static_cast<double>(setup.total_flops) / (result.median_ms * 1e6);

    runner.free_buffers();
    setup.free();

    return {tile, M, N, K, num_problems, result.median_ms, gflops};
}

// ---------------------------------------------------------------------------
// All tile shapes to benchmark
// ---------------------------------------------------------------------------

static const TileShape ALL_TILES[] = {
    TileShape::Tile_128x128,
    TileShape::Tile_64x64,
    TileShape::Tile_32x32,
    TileShape::Tile_256x32,
    TileShape::Tile_32x256,
    TileShape::Tile_128x16,
    TileShape::Tile_16x128,
    TileShape::Tile_64x8,
    TileShape::Tile_8x64,
};
static constexpr int NUM_TILES = sizeof(ALL_TILES) / sizeof(ALL_TILES[0]);

// ---------------------------------------------------------------------------
// Sweep mode: for each tile, run its "sweet spot" shapes + shared shapes
// ---------------------------------------------------------------------------

struct ShapeConfig {
    int M, N, K;
    const char* label;
};

template <typename T>
void run_sweep(int num_problems, int warmup, int repeat, cudaStream_t stream) {
    // Shapes designed to stress different aspect ratios.
    // Each tile gets tested on tile-aligned shapes AND realistic shapes.
    const ShapeConfig shapes[] = {
        // Square shapes
        {128, 128, 128, "128x128x128"},
        {256, 256, 128, "256x256x128"},
        {512, 512, 128, "512x512x128"},
        {64,  64,  128, "64x64x128"},
        {32,  32,  128, "32x32x128"},

        // Tall-skinny (M >> N)
        {256,  8,  128, "256x8x128"},
        {512,  8,  128, "512x8x128"},
        {128, 16,  128, "128x16x128"},
        {256, 16,  128, "256x16x128"},
        {256, 32,  128, "256x32x128"},
        {512, 32,  128, "512x32x128"},
        {1024, 8,  128, "1024x8x128"},
        {1024, 16, 128, "1024x16x128"},
        {1024, 32, 128, "1024x32x128"},

        // Wide-skinny (N >> M)
        {8,  256,  128, "8x256x128"},
        {8,  512,  128, "8x512x128"},
        {16, 128,  128, "16x128x128"},
        {16, 256,  128, "16x256x128"},
        {32, 256,  128, "32x256x128"},
        {32, 512,  128, "32x512x128"},
        {8,  1024, 128, "8x1024x128"},
        {16, 1024, 128, "16x1024x128"},
        {32, 1024, 128, "32x1024x128"},

        // Intermediate aspect ratios
        {128, 32,  128, "128x32x128"},
        {64,  16,  128, "64x16x128"},
        {32,  128, 128, "32x128x128"},
        {16,  64,  128, "16x64x128"},

        // Larger K
        {128, 128, 256, "128x128x256"},
        {256,  32, 256, "256x32x256"},
        {32,  256, 256, "32x256x256"},
        {128,  16, 256, "128x16x256"},
        {16,  128, 256, "16x128x256"},
        {64,   8,  256, "64x8x256"},
        {8,   64,  256, "8x64x256"},
    };
    const int num_shapes = sizeof(shapes) / sizeof(shapes[0]);

    const char* dtype_name = std::is_same<T, double>::value ? "f64" : "f32";
    printf("\n=== CUTLASS Tile Shape Efficiency Benchmark (%s) ===\n", dtype_name);
    printf("num_problems=%d, warmup=%d, repeat=%d\n\n", num_problems, warmup, repeat);

    // Header
    printf("%-16s", "Shape (MxNxK)");
    for (int t = 0; t < NUM_TILES; ++t) {
        printf("  %10s", tile_shape_name(ALL_TILES[t]));
    }
    printf("      best_tile\n");
    for (int i = 0; i < 16 + NUM_TILES * 12 + 16; ++i) printf("-");
    printf("\n");

    // Track per-tile peak GFLOP/s across all shapes
    double tile_peak[NUM_TILES] = {};

    for (int si = 0; si < num_shapes; ++si) {
        const auto& shape = shapes[si];
        printf("%-16s", shape.label);

        double best_gflops = 0;
        int best_tile_idx = 0;
        double gflops_arr[NUM_TILES];

        for (int t = 0; t < NUM_TILES; ++t) {
            auto r = bench_tile<T>(ALL_TILES[t], shape.M, shape.N, shape.K,
                                num_problems, warmup, repeat, stream);
            gflops_arr[t] = r.gflops;
            if (r.gflops > best_gflops) {
                best_gflops = r.gflops;
                best_tile_idx = t;
            }
            if (r.gflops > tile_peak[t]) tile_peak[t] = r.gflops;
        }

        for (int t = 0; t < NUM_TILES; ++t) {
            if (gflops_arr[t] <= 0) {
                printf("  %10s", "FAIL");
            } else {
                printf("  %10.1f", gflops_arr[t]);
            }
        }
        printf("  %12s\n", tile_shape_name(ALL_TILES[best_tile_idx]));
    }

    // Summary: peak GFLOP/s and efficiency ratios
    printf("\n=== Per-Tile Peak GFLOP/s ===\n");
    double max_peak = *std::max_element(tile_peak, tile_peak + NUM_TILES);
    printf("%-12s  %10s  %10s\n", "Tile", "Peak GF/s", "Efficiency");
    for (int i = 0; i < 36; ++i) printf("-");
    printf("\n");
    for (int t = 0; t < NUM_TILES; ++t) {
        float eff = max_peak > 0 ? static_cast<float>(tile_peak[t] / max_peak) : 0.0f;
        printf("%-12s  %10.1f  %10.3f\n",
               tile_shape_name(ALL_TILES[t]), tile_peak[t], eff);
    }
    printf("\nNormalized to peak (128x128 = %.1f GFLOP/s)\n", tile_peak[0]);
    printf("\n=== Efficiency relative to 128x128 peak ===\n");
    for (int t = 0; t < NUM_TILES; ++t) {
        float eff = tile_peak[0] > 0 ? static_cast<float>(tile_peak[t] / tile_peak[0]) : 0.0f;
        printf("  {TileShape::Tile_%-10s %3d, %3d, %.2ff},\n",
               (std::string(tile_shape_name(ALL_TILES[t])) + ",").c_str(),
               tile_m(ALL_TILES[t]), tile_n(ALL_TILES[t]), eff);
    }
}

// ---------------------------------------------------------------------------
// Default mode: one aligned problem shape, all tiles, sweep batch counts
// ---------------------------------------------------------------------------

template <typename T>
void run_default(int M, int N, int K, int num_problems, int warmup, int repeat,
                 cudaStream_t stream)
{
    const int counts[] = {1, 4, 16, 64, 256, 512, 1024};
    const int num_counts = sizeof(counts) / sizeof(counts[0]);

    bool single_count = (num_problems > 0);  // -1 = sweep

    const char* dtype_name = std::is_same<T, double>::value ? "f64" : "f32";
    printf("\n=== Tile Efficiency (%s): %dx%dx%d", dtype_name, M, N, K);
    if (single_count) printf(", %d problems", num_problems);
    printf(" ===\n\n");

    // Header
    printf("%-8s", "#probs");
    for (int t = 0; t < NUM_TILES; ++t)
        printf("  %10s", tile_shape_name(ALL_TILES[t]));
    printf("\n");
    for (int i = 0; i < 8 + NUM_TILES * 12; ++i) printf("-");
    printf("\n");

    // Collect peak GFLOP/s per tile across all batch counts
    double tile_peak[NUM_TILES] = {};

    auto run_row = [&](int np) {
        printf("%-8d", np);
        for (int t = 0; t < NUM_TILES; ++t) {
            auto r = bench_tile<T>(ALL_TILES[t], M, N, K, np, warmup, repeat, stream);
            if (r.gflops > 0) {
                printf("  %10.1f", r.gflops);
                if (r.gflops > tile_peak[t]) tile_peak[t] = r.gflops;
            } else {
                printf("  %10s", "FAIL");
            }
        }
        printf("\n");
    };

    if (single_count) {
        run_row(num_problems);
    } else {
        for (int ci = 0; ci < num_counts; ++ci)
            run_row(counts[ci]);
    }

    // Summary
    printf("\n=== Efficiency (relative to 128x128 peak) ===\n");
    double ref = tile_peak[0];  // 128x128
    for (int t = 0; t < NUM_TILES; ++t) {
        float eff = ref > 0 ? static_cast<float>(tile_peak[t] / ref) : 0.0f;
        printf("  %-12s  peak=%8.1f GFLOP/s  eff=%.3f\n",
               tile_shape_name(ALL_TILES[t]), tile_peak[t], eff);
    }
}

// ---------------------------------------------------------------------------
// Problem-count sweep: how does efficiency change with grouped GEMM size?
// ---------------------------------------------------------------------------

template <typename T>
void run_count_sweep(int warmup, int repeat, cudaStream_t stream) {
    const int counts[] = {4, 16, 64, 256, 512, 1024, 2048};
    const int num_counts = sizeof(counts) / sizeof(counts[0]);

    struct TestShape { int M, N, K; const char* label; };
    const TestShape shapes[] = {
        {256, 256, 128, "256x256"},
        {512, 512, 128, "512x512"},
        {192, 192, 128, "192x192"},  // 64x64 has padding advantage
        {600, 600, 128, "600x600"},  // large, ~equal padding
        {128,  16, 128, "128x16"},   // skinny
        {256,  32, 128, "256x32"},   // intermediate
    };

    const char* dtype_name = std::is_same<T, double>::value ? "f64" : "f32";
    printf("\n=== Problem-Count Sweep (%s): GFLOP/s by #problems ===\n\n", dtype_name);

    for (const auto& shape : shapes) {
        printf("--- %s (K=%d) ---\n", shape.label, shape.K);
        printf("%-8s", "#probs");
        for (int t = 0; t < NUM_TILES; ++t) {
            printf("  %10s", tile_shape_name(ALL_TILES[t]));
        }
        printf("\n");

        for (int ci = 0; ci < num_counts; ++ci) {
            int np = counts[ci];
            printf("%-8d", np);
            for (int t = 0; t < NUM_TILES; ++t) {
                auto r = bench_tile<T>(ALL_TILES[t], shape.M, shape.N, shape.K,
                                    np, warmup, repeat, stream);
                if (r.gflops > 0)
                    printf("  %10.1f", r.gflops);
                else
                    printf("  %10s", "FAIL");
            }
            printf("\n");
        }
        printf("\n");
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int warmup = 3;
    int repeat = 5;
    int num_problems = -1;  // -1 = sweep batch counts
    int M = 1024, N = 1024, K = 128;
    bool sweep = false;
    bool count_sweep = false;
    bool use_f64 = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (strcmp(argv[i], "--num-problems") == 0 && i + 1 < argc) num_problems = atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape") == 0 && i + 1 < argc) {
            // Parse MxNxK or MxN (K kept at default)
            ++i;
            if (sscanf(argv[i], "%dx%dx%d", &M, &N, &K) < 2) {
                fprintf(stderr, "Invalid --shape format, expected MxNxK or MxN\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
            ++i;
            if (strcmp(argv[i], "f64") == 0 || strcmp(argv[i], "double") == 0) {
                use_f64 = true;
            } else if (strcmp(argv[i], "f32") == 0 || strcmp(argv[i], "float") == 0) {
                use_f64 = false;
            } else {
                fprintf(stderr, "Unknown --dtype '%s', expected f32 or f64\n", argv[i]);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--sweep") == 0) sweep = true;
        else if (strcmp(argv[i], "--count-sweep") == 0) count_sweep = true;
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n\n", argv[0]);
            printf("Default: all tiles on one aligned shape, sweep batch counts\n");
            printf("  --shape MxNxK     Problem shape (default 1024x1024x128)\n");
            printf("  --num-problems N  Fixed batch count (default: sweep 1..1024)\n");
            printf("  --warmup N        Warmup iterations (default 3)\n");
            printf("  --repeat N        Timed iterations (default 5)\n");
            printf("  --dtype f32|f64   Element type (default f32)\n");
            printf("\nModes:\n");
            printf("  --sweep           M/N sweep across all tiles\n");
            printf("  --count-sweep     Batch-count sweep on multiple shapes\n");
            return 0;
        }
    }

    // Print GPU info
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    printf("GPU: %s (SM %d.%d, %d SMs)\n", props.name, props.major, props.minor,
           props.multiProcessorCount);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    if (use_f64) {
        if (count_sweep) {
            run_count_sweep<double>(warmup, repeat, stream);
        } else if (sweep) {
            int np = num_problems > 0 ? num_problems : 512;
            run_sweep<double>(np, warmup, repeat, stream);
        } else {
            run_default<double>(M, N, K, num_problems > 0 ? num_problems : -1,
                        warmup, repeat, stream);
        }
    } else {
        if (count_sweep) {
            run_count_sweep<float>(warmup, repeat, stream);
        } else if (sweep) {
            int np = num_problems > 0 ? num_problems : 512;
            run_sweep<float>(np, warmup, repeat, stream);
        } else {
            run_default<float>(M, N, K, num_problems > 0 ? num_problems : -1,
                        warmup, repeat, stream);
        }
    }

    cudaStreamDestroy(stream);
    return 0;
}
