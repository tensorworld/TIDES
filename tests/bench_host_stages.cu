// Host-side preprocessing benchmark for analysis pipeline stages.
// Measures push (with MNSplitter), sort_packets, pack_batches on synthetic
// GemmProblemDesc arrays. No tensor files or GPU kernels needed.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "general_sparse/pipeline_descs.hpp"
#include "general_sparse/analysis/pipeline.hpp"
#include "general_sparse/analysis/dispatchers.hpp"
#include "general_sparse/analysis/passes/mn_split.hpp"
#include "general_sparse/analysis/passes/k_split.hpp"
#include "general_sparse/analysis/passes/compose.hpp"
#include "general_sparse/kernels/small/warp_gemm.cuh"

using namespace bsptc;

// Local descriptor type for benchmarks: MNSplitter/KSplitter only need F_Strides.
using BenchDesc = Desc<F_Ptrs, F_MNK, F_Strides>;

// Trait-based helpers (moved from pipeline_descs.hpp; only used here).
template <typename D>
bool sort_by_K_then_MN(const D& a, const D& b) {
    if (a.K != b.K) return a.K > b.K;
    return a.M * a.N < b.M * b.N;
}

template <typename D>
int mn_work(const D& d) {
    return d.M * d.N;
}

// Host-side batch packing constants (previously in v2_batched_gemm.cuh)
constexpr int TARGET_MN_PER_BATCH = 512;
constexpr int MAX_DESCS_PER_BATCH = 64;
constexpr int MIN_DESCS_FOR_BATCHING = 4;

// ---------------------------------------------------------------------------
// CPU timing harness
// ---------------------------------------------------------------------------

struct BenchResult { double median_us, min_us, max_us; };

template <typename Fn>
BenchResult cpu_bench(Fn fn, int warmup, int repeat) {
    for (int i = 0; i < warmup; i++) fn();

    std::vector<double> times(repeat);
    for (int i = 0; i < repeat; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    return {times[repeat / 2], times[0], times[repeat - 1]};
}

// ---------------------------------------------------------------------------
// Synthetic problem generation
// ---------------------------------------------------------------------------

struct ProblemSpec {
    int num_problems;
    int batch_size;
    int M, N, K;
};

static std::vector<GemmProblemDesc> make_problems(const ProblemSpec& spec) {
    std::vector<GemmProblemDesc> problems(spec.num_problems);
    int64_t offset = 0;
    for (int i = 0; i < spec.num_problems; i++) {
        auto& p = problems[i];
        p.pair_idx = i;
        p.block_idx_A = i;
        p.block_idx_B = i;
        p.M = spec.M;
        p.N = spec.N;
        p.K = spec.K;
        p.batch_size = spec.batch_size;
        p.lda = spec.K;
        p.ldb = spec.N;
        p.ldc = spec.N;
        p.offset_A = offset;
        p.offset_B = offset;
        p.offset_C = offset;
        offset += spec.M * spec.N * spec.batch_size;
    }
    return problems;
}

static std::vector<BlockPairInfo> make_pairs(int num, int64_t output_volume) {
    std::vector<BlockPairInfo> pairs(num);
    for (int i = 0; i < num; i++) {
        pairs[i].output_offset = i * output_volume;
        pairs[i].output_volume = output_volume;
    }
    return pairs;
}

// ---------------------------------------------------------------------------
// Benchmark: expand small (push through pipeline with MNSplitter)
// ---------------------------------------------------------------------------

static void bench_expand_small(int warmup, int repeat) {
    printf("\n=== expand small (pipeline + MNSplitter) ===\n");
    printf("%-8s %5s %5s %5s %5s %10s %10s %10s %8s\n",
           "NProbs", "Batch", "M", "N", "K", "Median(us)", "Min(us)", "Max(us)", "NDescs");
    printf("%-8s %5s %5s %5s %5s %10s %10s %10s %8s\n",
           "------", "-----", "-----", "-----", "-----", "----------", "----------", "----------", "------");

    ProblemSpec specs[] = {
        {100,   1, 16, 16, 16},
        {1000,  1, 16, 16, 16},
        {5000,  1, 16, 16, 16},
        {20000, 1, 16, 16, 16},
        {1000,  4, 16, 16, 16},
        {1000,  1, 64, 64, 64},
        {1000,  1, 700, 3, 32},   // triggers M split
        {1000,  1, 4, 700, 32},   // triggers N split
        {5000,  8, 8, 8, 32},
    };

    float* dummy = reinterpret_cast<float*>(0x1000);

    for (auto& spec : specs) {
        auto problems = make_problems(spec);
        int64_t out_vol = spec.M * spec.N * spec.batch_size;
        auto pairs = make_pairs(spec.num_problems, out_vol);

        int ndescs = 0;
        auto r = cpu_bench([&]() {
            auto pipe = make_pipeline<BenchDesc>()
                .transform(MNSplitter{512, true, false, sizeof(float)})
                .build();

            for (const auto& prob : problems) {
                for (int64_t b = 0; b < prob.batch_size; ++b) {
                    const float* base_A = dummy + prob.offset_A + b * prob.M * prob.K;
                    const float* base_B = dummy + prob.offset_B + b * prob.K * prob.N;
                    float* base_C = dummy +
                           pairs[prob.pair_idx].output_offset + b * prob.M * prob.N;
                    auto sd = make_stride_desc<BenchDesc>(base_A, base_B, base_C,
                                  prob.M, prob.N, prob.K, prob.K, prob.N, prob.N);
                    pipe.push(sd, prob.offset_A, prob.offset_B);
                }
            }
            ndescs = static_cast<int>(pipe.collect().size());
        }, warmup, repeat);

        printf("%8d %5d %5d %5d %5d %10.1f %10.1f %10.1f %8d\n",
               spec.num_problems, spec.batch_size, spec.M, spec.N, spec.K,
               r.median_us, r.min_us, r.max_us, ndescs);
    }
}

// ---------------------------------------------------------------------------
// Benchmark: sort_packets (BenchDesc)
// ---------------------------------------------------------------------------

static void bench_sort_small(int warmup, int repeat) {
    printf("\n=== sort_packets (BenchDesc) ===\n");
    printf("%-8s %10s %10s %10s\n",
           "NDescs", "Median(us)", "Min(us)", "Max(us)");
    printf("%-8s %10s %10s %10s\n",
           "------", "----------", "----------", "----------");

    int counts[] = {100, 500, 1000, 5000, 20000};

    for (int n : counts) {
        using Pkt = Packet<BenchDesc>;
        std::vector<Pkt> base(n);
        for (int i = 0; i < n; i++) {
            base[i].desc.M = 4 + (i * 7) % 60;
            base[i].desc.N = 4 + (i * 13) % 60;
            base[i].desc.K = 4 + (i * 3) % 128;
            base[i].offset_A = i * 100;
            base[i].offset_B = i * 200;
        }

        auto r = cpu_bench([&]() {
            auto copy = base;
            sort_packets(copy, sort_by_K_then_MN<BenchDesc>);
        }, warmup, repeat);

        printf("%8d %10.1f %10.1f %10.1f\n", n, r.median_us, r.min_us, r.max_us);
    }
}

// ---------------------------------------------------------------------------
// Benchmark: pack_batches (BenchDesc)
// ---------------------------------------------------------------------------

static void bench_pack_small(int warmup, int repeat) {
    printf("\n=== pack_batches (BenchDesc) ===\n");
    printf("%-8s %10s %10s %10s %8s\n",
           "NDescs", "Median(us)", "Min(us)", "Max(us)", "NBatch");
    printf("%-8s %10s %10s %10s %8s\n",
           "------", "----------", "----------", "----------", "------");

    int counts[] = {100, 500, 1000, 5000, 20000};

    for (int n : counts) {
        using Pkt = Packet<BenchDesc>;
        std::vector<Pkt> packets(n);
        for (int i = 0; i < n; i++) {
            packets[i].desc.M = 4 + (i * 7) % 60;
            packets[i].desc.N = 4 + (i * 13) % 60;
            packets[i].desc.K = 4 + (i * 3) % 128;
            packets[i].offset_A = i * 100;
            packets[i].offset_B = i * 200;
        }
        sort_packets(packets, sort_by_K_then_MN<BenchDesc>);

        int nbatch = 0;
        auto r = cpu_bench([&]() {
            auto ranges = pack_batches(packets,
                mn_work<BenchDesc>,
                TARGET_MN_PER_BATCH, MAX_DESCS_PER_BATCH);
            nbatch = static_cast<int>(ranges.size());
        }, warmup, repeat);

        printf("%8d %10.1f %10.1f %10.1f %8d\n", n, r.median_us, r.min_us, r.max_us, nbatch);
    }
}

// ---------------------------------------------------------------------------
// Benchmark: full pipeline (expand → sort → pack) for small tier
// ---------------------------------------------------------------------------

static void bench_full_small_pipeline(int warmup, int repeat) {
    printf("\n=== Full small pipeline: expand → sort → pack ===\n");
    printf("%-8s %5s %5s %5s %5s %10s %10s %10s %8s %8s\n",
           "NProbs", "Batch", "M", "N", "K", "Median(us)", "Min(us)", "Max(us)", "NDescs", "NBatch");
    printf("%-8s %5s %5s %5s %5s %10s %10s %10s %8s %8s\n",
           "------", "-----", "-----", "-----", "-----", "----------", "----------", "----------", "------", "------");

    ProblemSpec specs[] = {
        {100,   1, 16, 16, 16},
        {1000,  1, 16, 16, 16},
        {5000,  1, 16, 16, 16},
        {20000, 1, 16, 16, 16},
        {1000,  4, 16, 16, 16},
        {5000,  8, 8, 8, 32},
        {1000,  1, 700, 3, 32},
    };

    float* dummy = reinterpret_cast<float*>(0x1000);

    for (auto& spec : specs) {
        auto problems = make_problems(spec);
        int64_t out_vol = spec.M * spec.N * spec.batch_size;
        auto pairs = make_pairs(spec.num_problems, out_vol);

        int ndescs = 0, nbatch = 0;
        auto r = cpu_bench([&]() {
            // Use collect-only pipeline (no dispatcher) so this bench stays
            // CPU-only. Production code uses build(dispatcher) + finish().
            auto pipe = make_pipeline<BenchDesc>()
                .transform(MNSplitter{512, true, false, sizeof(float)})
                .build();

            for (const auto& prob : problems) {
                for (int64_t b = 0; b < prob.batch_size; ++b) {
                    const float* base_A = dummy + prob.offset_A + b * prob.M * prob.K;
                    const float* base_B = dummy + prob.offset_B + b * prob.K * prob.N;
                    float* base_C = dummy +
                           pairs[prob.pair_idx].output_offset + b * prob.M * prob.N;
                    auto sd = make_stride_desc<BenchDesc>(base_A, base_B, base_C,
                                  prob.M, prob.N, prob.K, prob.K, prob.N, prob.N);
                    pipe.push(sd, prob.offset_A, prob.offset_B);
                }
            }

            auto& packets = pipe.collect();
            ndescs = static_cast<int>(packets.size());
            nbatch = 0;
            if (ndescs >= MIN_DESCS_FOR_BATCHING) {
                sort_packets(packets, sort_by_K_then_MN<BenchDesc>);
                auto ranges = pack_batches(packets,
                    mn_work<BenchDesc>,
                    TARGET_MN_PER_BATCH, MAX_DESCS_PER_BATCH);
                nbatch = static_cast<int>(ranges.size());
            }
        }, warmup, repeat);

        printf("%8d %5d %5d %5d %5d %10.1f %10.1f %10.1f %8d %8d\n",
               spec.num_problems, spec.batch_size, spec.M, spec.N, spec.K,
               r.median_us, r.min_us, r.max_us, ndescs, nbatch);
    }
}

// ---------------------------------------------------------------------------
// Benchmark: expand medium (push through pipeline, no MN split)
// ---------------------------------------------------------------------------

static void bench_expand_medium(int warmup, int repeat) {
    printf("\n=== expand medium (pipeline) ===\n");
    printf("%-8s %5s %5s %5s %5s %10s %10s %10s %8s\n",
           "NProbs", "Batch", "M", "N", "K", "Median(us)", "Min(us)", "Max(us)", "NDescs");
    printf("%-8s %5s %5s %5s %5s %10s %10s %10s %8s\n",
           "------", "-----", "-----", "-----", "-----", "----------", "----------", "----------", "------");

    ProblemSpec specs[] = {
        {100,   1, 32, 32, 16},
        {1000,  1, 32, 32, 16},
        {5000,  1, 32, 32, 16},
        {1000,  4, 16, 16, 32},
        {5000,  8, 8, 8, 64},
    };

    float* dummy = reinterpret_cast<float*>(0x1000);

    for (auto& spec : specs) {
        auto problems = make_problems(spec);
        int64_t out_vol = spec.M * spec.N * spec.batch_size;
        auto pairs = make_pairs(spec.num_problems, out_vol);

        int ndescs = 0;
        auto r = cpu_bench([&]() {
            auto pipe = make_pipeline<BenchDesc>().build();

            for (const auto& prob : problems) {
                for (int64_t b = 0; b < prob.batch_size; ++b) {
                    const float* base_A = dummy + prob.offset_A + b * prob.M * prob.K;
                    const float* base_B = dummy + prob.offset_B + b * prob.K * prob.N;
                    float* base_C = dummy +
                           pairs[prob.pair_idx].output_offset + b * prob.M * prob.N;

                    auto sd = make_stride_desc<BenchDesc>(base_A, base_B, base_C,
                                  prob.M, prob.N, prob.K, prob.lda, prob.ldb, prob.N);
                    pipe.push(sd, prob.offset_A, prob.offset_B);
                }
            }
            ndescs = static_cast<int>(pipe.collect().size());
        }, warmup, repeat);

        printf("%8d %5d %5d %5d %5d %10.1f %10.1f %10.1f %8d\n",
               spec.num_problems, spec.batch_size, spec.M, spec.N, spec.K,
               r.median_us, r.min_us, r.max_us, ndescs);
    }
}

// ---------------------------------------------------------------------------
// Benchmark: upload_packets (measures cudaMalloc + cudaMemcpyAsync overhead)
// ---------------------------------------------------------------------------

static void bench_upload(int warmup, int repeat) {
    printf("\n=== upload_packets (BenchDesc) ===\n");
    printf("%-8s %10s %10s %10s %10s\n",
           "NDescs", "Median(us)", "Min(us)", "Max(us)", "Bytes");
    printf("%-8s %10s %10s %10s %10s\n",
           "------", "----------", "----------", "----------", "----------");

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int counts[] = {100, 500, 1000, 5000, 20000};

    for (int n : counts) {
        using Pkt = Packet<BenchDesc>;
        std::vector<Pkt> packets(n);
        for (int i = 0; i < n; i++) {
            packets[i].desc.M = 16;
            packets[i].desc.N = 16;
            packets[i].desc.K = 16;
        }

        BenchDesc* d_ptr = nullptr;
        auto r = cpu_bench([&]() {
            if (d_ptr) cudaFree(d_ptr);
            auto result = upload_packets(packets, {}, stream);
            d_ptr = result.d_descs;
            cudaStreamSynchronize(stream);
        }, warmup, repeat);

        if (d_ptr) cudaFree(d_ptr);

        printf("%8d %10.1f %10.1f %10.1f %10zu\n",
               n, r.median_us, r.min_us, r.max_us, n * sizeof(BenchDesc));
    }

    cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
// Benchmark: compose(MNSplitter, KSplitter) expansion
// ---------------------------------------------------------------------------

static void bench_expand_k_split(int warmup, int repeat) {
    printf("\n=== expand small (compose(MNSplitter, KSplitter)) ===\n");
    printf("%-8s %5s %5s %5s %5s %8s %10s %10s %10s %8s %6s\n",
           "NProbs", "Batch", "M", "N", "K", "KThresh", "Median(us)", "Min(us)", "Max(us)", "NDescs", "Ratio");
    printf("%-8s %5s %5s %5s %5s %8s %10s %10s %10s %8s %6s\n",
           "------", "-----", "-----", "-----", "-----", "--------", "----------", "----------", "----------", "------", "-----");

    struct KSplitSpec {
        ProblemSpec prob;
        int k_threshold;
    };

    KSplitSpec specs[] = {
        // K not split (K <= threshold)
        {{1000, 1, 16, 16, 64},  128},
        // K split into 2 chunks
        {{1000, 1, 16, 16, 256}, 128},
        // K split into 4 chunks
        {{1000, 1, 16, 16, 512}, 128},
        // Large K with small MN
        {{1000, 1, 4, 4, 1024},  128},
        // MN split + K split combined
        {{1000, 1, 700, 3, 512}, 128},
        // Different threshold
        {{1000, 1, 16, 16, 512}, 256},
        // Many problems, moderate K split
        {{5000, 1, 8, 8, 256},   128},
        // Batch + K split
        {{1000, 4, 8, 8, 256},   128},
    };

    float* dummy = reinterpret_cast<float*>(0x1000);

    for (auto& spec : specs) {
        auto problems = make_problems(spec.prob);
        int64_t out_vol = spec.prob.M * spec.prob.N * spec.prob.batch_size;
        auto pairs = make_pairs(spec.prob.num_problems, out_vol);

        int ndescs = 0;
        int ndescs_base = 0;
        auto r = cpu_bench([&]() {
            auto pipe = make_pipeline<BenchDesc>()
                .transform(compose(
                    MNSplitter{512, true, true, sizeof(float)},
                    KSplitter{static_cast<dim_t>(spec.k_threshold), true, true, sizeof(float)}))
                .build();

            for (const auto& prob : problems) {
                for (int64_t b = 0; b < prob.batch_size; ++b) {
                    const float* base_A = dummy + prob.offset_A + b * prob.M * prob.K;
                    const float* base_B = dummy + prob.offset_B + b * prob.K * prob.N;
                    float* base_C = dummy +
                           pairs[prob.pair_idx].output_offset + b * prob.M * prob.N;
                    auto sd = make_stride_desc<BenchDesc>(base_A, base_B, base_C,
                                  prob.M, prob.N, prob.K, prob.K, prob.N, prob.N);
                    pipe.push(sd, prob.offset_A, prob.offset_B);
                }
            }
            ndescs = static_cast<int>(pipe.collect().size());
        }, warmup, repeat);

        // Compute baseline (MN split only) for ratio
        {
            auto pipe = make_pipeline<BenchDesc>()
                .transform(MNSplitter{512, true, true, sizeof(float)})
                .build();
            for (const auto& prob : problems) {
                for (int64_t b = 0; b < prob.batch_size; ++b) {
                    const float* base_A = dummy + prob.offset_A + b * prob.M * prob.K;
                    const float* base_B = dummy + prob.offset_B + b * prob.K * prob.N;
                    float* base_C = dummy +
                           pairs[prob.pair_idx].output_offset + b * prob.M * prob.N;
                    auto sd = make_stride_desc<BenchDesc>(base_A, base_B, base_C,
                                  prob.M, prob.N, prob.K, prob.K, prob.N, prob.N);
                    pipe.push(sd, prob.offset_A, prob.offset_B);
                }
            }
            ndescs_base = static_cast<int>(pipe.collect().size());
        }

        printf("%8d %5d %5d %5d %5d %8d %10.1f %10.1f %10.1f %8d %5.1fx\n",
               spec.prob.num_problems, spec.prob.batch_size,
               spec.prob.M, spec.prob.N, spec.prob.K, spec.k_threshold,
               r.median_us, r.min_us, r.max_us, ndescs,
               ndescs_base > 0 ? static_cast<double>(ndescs) / ndescs_base : 0.0);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int warmup = 5, repeat = 20;
    std::string filter;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc)
            repeat = atoi(argv[++i]);
        else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc)
            filter = argv[++i];
        else {
            fprintf(stderr, "Usage: %s [--warmup N] [--repeat N] [--filter expand|sort|pack|full|medium|upload|ksplit]\n",
                    argv[0]);
            return 1;
        }
    }

    printf("Host Preprocessing Stage Benchmarks (warmup=%d, repeat=%d)\n", warmup, repeat);

    bool all = filter.empty();

    if (all || filter.find("expand") != std::string::npos)
        bench_expand_small(warmup, repeat);
    if (all || filter.find("sort") != std::string::npos)
        bench_sort_small(warmup, repeat);
    if (all || filter.find("pack") != std::string::npos)
        bench_pack_small(warmup, repeat);
    if (all || filter.find("full") != std::string::npos)
        bench_full_small_pipeline(warmup, repeat);
    if (all || filter.find("medium") != std::string::npos)
        bench_expand_medium(warmup, repeat);
    if (all || filter.find("upload") != std::string::npos)
        bench_upload(warmup, repeat);
    if (all || filter.find("ksplit") != std::string::npos)
        bench_expand_k_split(warmup, repeat);

    printf("\nDone.\n");
    return 0;
}
