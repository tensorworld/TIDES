// Per-kernel benchmark for medium GEMM (volume kernel).
// Modes:
//   Tensor mode (default): loads .bs files, runs pipeline_setup, benchmarks volume GEMM.
//   Synthetic mode (--synthetic): creates uniform-sized blocks for controlled sweeps.
//
// Both modes use pipeline_setup + stage functions from pipeline_stages.hpp,
// so the benchmark exercises the exact production code paths.
//
// Usage:
//   ./bench_medium --filter <substr> [--warmup N] [--repeat N] [--dtype f64]
//   ./bench_medium --synthetic [--einsum "jik,jkl->il"] [--num-descs 256]
//                  [--warmup N] [--repeat N]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "bs_parser.hpp"
#include "einsum.hpp"
#include "general_sparse/gpu_sparse_input.hpp"
#include "general_sparse/pipeline.cuh"

using namespace bsptc;
namespace fs = std::filesystem;

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
// Info file parser (tensor mode)
// ---------------------------------------------------------------------------

struct ContractionInfo {
    std::string einsum;
    int A_blocks, B_blocks;
    int A_rank, B_rank;
};

ContractionInfo parse_info_file(const std::string& path) {
    ContractionInfo info{};
    std::ifstream f(path);
    if (!f.is_open()) return info;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("einsum:", 0) == 0) info.einsum = line.substr(8);
        else if (line.rfind("A_blocks:", 0) == 0) info.A_blocks = std::stoi(line.substr(10));
        else if (line.rfind("B_blocks:", 0) == 0) info.B_blocks = std::stoi(line.substr(10));
        else if (line.rfind("A_rank:", 0) == 0) info.A_rank = std::stoi(line.substr(8));
        else if (line.rfind("B_rank:", 0) == 0) info.B_rank = std::stoi(line.substr(8));
    }
    while (!info.einsum.empty() && (info.einsum.back() == ' ' || info.einsum.back() == '\n'))
        info.einsum.pop_back();
    while (!info.einsum.empty() && info.einsum.front() == ' ')
        info.einsum.erase(info.einsum.begin());
    return info;
}

// ---------------------------------------------------------------------------
// Discover tensor cases (tensor mode)
// ---------------------------------------------------------------------------

struct TensorCase {
    std::string name;
    std::string dir_path;
    ContractionInfo info;
};

std::vector<TensorCase> discover_cases(const std::string& filter) {
    std::vector<TensorCase> cases;
    std::string base_dir = std::string(getenv("HOME")) +
        "/data/block_sparse_dev/generated_tensors/cutensor_converted";

    for (auto& entry : fs::directory_iterator(base_dir)) {
        if (!entry.is_directory()) continue;
        std::string name = entry.path().filename().string();
        if (!filter.empty() && name.find(filter) == std::string::npos) continue;

        std::string info_path = entry.path().string() + "/" + name + "_info.txt";
        if (!fs::exists(info_path)) continue;

        auto info = parse_info_file(info_path);
        if (info.einsum.empty()) continue;

        cases.push_back({name, entry.path().string(), info});
    }

    std::sort(cases.begin(), cases.end(),
              [](const TensorCase& a, const TensorCase& b) { return a.name < b.name; });
    return cases;
}

// ---------------------------------------------------------------------------
// Medium block shape statistics
// ---------------------------------------------------------------------------

struct ShapeStats {
    int count = 0;
    int min_M = INT_MAX, max_M = 0;
    int min_N = INT_MAX, max_N = 0;
    int min_K = INT_MAX, max_K = 0;
    int64_t total_flops = 0;

    void add(int M, int N, int K, int batch) {
        count += batch;
        min_M = std::min(min_M, M); max_M = std::max(max_M, M);
        min_N = std::min(min_N, N); max_N = std::max(max_N, N);
        min_K = std::min(min_K, K); max_K = std::max(max_K, K);
        total_flops += 2LL * M * N * K * batch;
    }

    void print() const {
        if (count == 0) { printf("  (no medium blocks)\n"); return; }
        printf("  %d problems, M=[%d,%d], N=[%d,%d], K=[%d,%d], %.3f GFLOP\n",
               count, min_M, max_M, min_N, max_N, min_K, max_K,
               static_cast<double>(total_flops) / 1e9);
    }
};

// ---------------------------------------------------------------------------
// Shared benchmark core: given a pipeline plan, benchmark medium GEMM
// ---------------------------------------------------------------------------

template <typename T>
void bench_medium_plan(PipelinePlan<T>& plan,
                       const GPUSparseInput<T>& /*A_gpu*/, const GPUSparseInput<T>& /*B_gpu*/,
                       const DeviceSparseInput<T>& /*A_dev*/, const DeviceSparseInput<T>& /*B_dev*/,
                       int warmup, int repeat, const char* label)
{
    cudaStream_t stream = plan.stream_medium;
    ShapeStats stats;
    for (const auto& p : plan.work.medium.problems)
        stats.add(p.M, p.N, p.K, p.batch_size);
    int64_t total_flops = stats.total_flops;

    // Benchmark medium GEMM (volume kernel — no transpose needed)
    auto result = gpu_bench([&]() {
        stage_medium_gemm(plan, stream);
    }, warmup, repeat, stream);

    float gemm_ms = result.median_ms;
    double gemm_gflops = static_cast<double>(total_flops) / (gemm_ms * 1e6);

    printf("  %s: %d descs, %.3f ms, %.1f GFLOP/s\n",
           label, plan.medium_vol_num_problems, gemm_ms, gemm_gflops);
}

// ---------------------------------------------------------------------------
// Tensor mode: run benchmark for one tensor case
// ---------------------------------------------------------------------------

template <typename T>
void run_tensor_case(const TensorCase& tc, int warmup, int repeat) {
    std::string fileA = tc.dir_path + "/" + tc.name + "_A.bs";
    std::string fileB = tc.dir_path + "/" + tc.name + "_B.bs";

    DataType dtype = std::is_same<T, double>::value ? DataType::FLOAT64 : DataType::FLOAT32;
    auto tensors = parse_tensor_files({fileA, fileB}, dtype);
    auto& A = tensors[0];
    auto& B = tensors[1];

    auto expr = parse_einsum(tc.info.einsum);
    std::vector<std::vector<int64_t>> shapes = {
        block_utils::compute_full_shape(A), block_utils::compute_full_shape(B)
    };
    validate_einsum(expr, shapes);

    EinsumIndexSets index_sets;
    create_einsum_index_sets(expr, index_sets);

    auto A_gpu = to_gpu_input<T>(A);
    auto B_gpu = to_gpu_input<T>(B);

    auto A_dev = copy_sparse_to_device(A_gpu);
    auto B_dev = copy_sparse_to_device(B_gpu);

    printf("\n%s  einsum=%s  A_blk=%d  B_blk=%d\n",
           tc.name.c_str(), tc.info.einsum.c_str(), tc.info.A_blocks, tc.info.B_blocks);

    std::string output_labels(expr.output_labels.begin(), expr.output_labels.end());

    PipelinePlan<T> plan;
    pipeline_setup<T>(plan, A_gpu, B_gpu, A_dev, B_dev, index_sets, output_labels);

    ShapeStats stats;
    for (const auto& p : plan.work.medium.problems)
        stats.add(p.M, p.N, p.K, p.batch_size);
    stats.print();
    if (stats.count == 0) {
        printf("  (skipping — no medium blocks)\n");
        return;
    }

    if (plan.medium_vol_num_problems == 0) {
        printf("  volume: N/A (pipeline did not build volume descs)\n");
    } else {
        bench_medium_plan(plan, A_gpu, B_gpu, A_dev, B_dev,
                          warmup, repeat, "volume");
    }
}

// ---------------------------------------------------------------------------
// Synthetic mode: helpers
// ---------------------------------------------------------------------------

// Split `total` into `ndims` factors as evenly as possible.
static std::vector<int> split_dim(int total, int ndims) {
    if (ndims <= 1) return {total};
    double root = std::pow(static_cast<double>(total), 1.0 / ndims);
    int base = std::max(1, static_cast<int>(root));
    std::vector<int> dims(ndims, base);
    int product = 1;
    for (int i = 0; i < ndims - 1; ++i) product *= dims[i];
    dims[ndims - 1] = std::max(1, total / product);
    return dims;
}

// Build a synthetic GPUSparseInput for one operand.
// All blocks have identical dims. Blocks vary in the `vary_dim` position.
template <typename T>
static GPUSparseInput<T> build_synthetic_gpu_input(
    const std::vector<char>& labels,
    const std::map<char, int>& label_size,
    int num_blocks,
    int vary_pos)  // which label position varies across blocks (-1 = none)
{
    dim_t rank = static_cast<dim_t>(labels.size());
    GPUSparseInput<T> gpu;
    gpu.rank = rank;
    gpu.num_blocks = static_cast<dim_t>(num_blocks);
    gpu.labels.assign(labels.begin(), labels.end());

    // block_dims: same shape for every block
    gpu.block_dims.resize(num_blocks * rank);
    int64_t block_vol = 1;
    for (dim_t d = 0; d < rank; ++d) {
        int sz = label_size.at(labels[d]);
        block_vol *= sz;
        for (int i = 0; i < num_blocks; ++i)
            gpu.block_dims[i * rank + d] = sz;
    }

    // block_offsets, block_volumes
    gpu.block_offsets.resize(num_blocks);
    gpu.block_volumes.resize(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        gpu.block_offsets[i] = static_cast<int64_t>(i) * block_vol;
        gpu.block_volumes[i] = block_vol;
    }
    gpu.total_values = static_cast<int64_t>(num_blocks) * block_vol;

    // block_coords: vary in vary_pos, 0 elsewhere
    gpu.block_coords.resize(num_blocks * rank, 0);
    if (vary_pos >= 0) {
        for (int i = 0; i < num_blocks; ++i)
            gpu.block_coords[i * rank + vary_pos] = i;
    }

    // block_sizes_flat, block_sizes_offsets, num_blocks_per_dim
    gpu.num_blocks_per_dim.resize(rank);
    gpu.block_sizes_offsets.resize(rank);
    int offset = 0;
    for (dim_t d = 0; d < rank; ++d) {
        gpu.block_sizes_offsets[d] = offset;
        int sz = label_size.at(labels[d]);
        if (d == vary_pos) {
            gpu.num_blocks_per_dim[d] = num_blocks;
            for (int i = 0; i < num_blocks; ++i)
                gpu.block_sizes_flat.push_back(sz);
            offset += num_blocks;
        } else {
            gpu.num_blocks_per_dim[d] = 1;
            gpu.block_sizes_flat.push_back(sz);
            offset += 1;
        }
    }

    // full_extents
    gpu.full_extents.resize(rank);
    for (dim_t d = 0; d < rank; ++d) {
        int sz = label_size.at(labels[d]);
        gpu.full_extents[d] = (d == vary_pos) ? static_cast<int64_t>(sz) * num_blocks
                                               : static_cast<int64_t>(sz);
    }

    // Fill with random data (parallelized with per-thread RNGs)
    gpu.values.resize(static_cast<size_t>(gpu.total_values));
    {
        size_t n = gpu.values.size();
        #pragma omp parallel
        {
            int tid = 0, nthreads = 1;
            #ifdef _OPENMP
            tid = omp_get_thread_num();
            nthreads = omp_get_num_threads();
            #endif
            std::mt19937 rng(42 + rank + tid * 1000003u);
            std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
            size_t lo = n * static_cast<size_t>(tid) / static_cast<size_t>(nthreads);
            size_t hi = n * static_cast<size_t>(tid + 1) / static_cast<size_t>(nthreads);
            for (size_t i = lo; i < hi; ++i)
                gpu.values[i] = static_cast<T>(dist(rng));
        }
    }

    return gpu;
}

// Build synthetic A/B inputs for a given einsum with uniform M/N/K per block.
// Creates num_pairs A blocks × 1 B block → num_pairs pairs.
template <typename T>
struct SyntheticInputs {
    GPUSparseInput<T> A_gpu, B_gpu;
    EinsumIndexSets index_sets;
    std::string output_labels;
    int actual_M, actual_N, actual_K;
};

template <typename T>
SyntheticInputs<T> build_synthetic_inputs(
    const std::string& einsum_str, int M, int N, int K, int num_pairs)
{
    auto expr = parse_einsum(einsum_str);
    EinsumIndexSets idx;
    create_einsum_index_sets(expr, idx);

    const auto& A_labels = expr.inputs[0].labels;
    const auto& B_labels = expr.inputs[1].labels;

    // Split M/N/K across sub-dimensions
    auto m_dims = split_dim(M, static_cast<int>(idx.free_indices_A.size()));
    auto n_dims = split_dim(N, static_cast<int>(idx.free_indices_B.size()));
    auto k_dims = split_dim(K, static_cast<int>(idx.contracted_indices.size()));

    // Build label→size map
    std::map<char, int> label_size;
    for (size_t i = 0; i < idx.free_indices_A.size(); ++i)
        label_size[idx.free_indices_A[i]] = m_dims[i];
    for (size_t i = 0; i < idx.free_indices_B.size(); ++i)
        label_size[idx.free_indices_B[i]] = n_dims[i];
    for (size_t i = 0; i < idx.contracted_indices.size(); ++i)
        label_size[idx.contracted_indices[i]] = k_dims[i];
    for (char c : idx.batch_indices)
        label_size[c] = 1;

    // A: vary in first free_A label position to get num_pairs blocks
    char vary_label = idx.free_indices_A[0];
    int vary_pos_A = -1;
    for (size_t d = 0; d < A_labels.size(); ++d)
        if (A_labels[d] == vary_label) { vary_pos_A = static_cast<int>(d); break; }

    auto A_gpu = build_synthetic_gpu_input<T>(
        std::vector<char>(A_labels.begin(), A_labels.end()),
        label_size, num_pairs, vary_pos_A);

    // B: 1 block (vary_label is free_A, doesn't appear in B)
    auto B_gpu = build_synthetic_gpu_input<T>(
        std::vector<char>(B_labels.begin(), B_labels.end()),
        label_size, 1, -1);

    // Compute actual M/N/K after split_dim rounding
    int actual_M = 1, actual_N = 1, actual_K = 1;
    for (int d : m_dims) actual_M *= d;
    for (int d : n_dims) actual_N *= d;
    for (int d : k_dims) actual_K *= d;

    std::string output_labels(expr.output_labels.begin(), expr.output_labels.end());

    SyntheticInputs<T> result;
    result.A_gpu = std::move(A_gpu);
    result.B_gpu = std::move(B_gpu);
    result.index_sets = std::move(idx);
    result.output_labels = std::move(output_labels);
    result.actual_M = actual_M;
    result.actual_N = actual_N;
    result.actual_K = actual_K;
    return result;
}

// ---------------------------------------------------------------------------
// Synthetic mode: run one benchmark point
// ---------------------------------------------------------------------------

template <typename T>
void bench_synthetic_point(const std::string& einsum_str, int M, int N, int K,
                           int num_descs, int warmup, int repeat) {
    auto synth = build_synthetic_inputs<T>(einsum_str, M, N, K, num_descs);
    M = synth.actual_M; N = synth.actual_N; K = synth.actual_K;

    if (M < 9 || N < 9) return;  // below volume threshold
    if (vol_tile_count(M, N) > VOL_MAX_TILES) return;  // above volume tile limit
    if (static_cast<int64_t>(M) * K + static_cast<int64_t>(N) * K > vol_threshold<T>()) return;

    int64_t total_flops = 2LL * M * N * K * num_descs;
    printf("  M=%3d N=%3d K=%3d:", M, N, K);

    auto A_dev = copy_sparse_to_device(synth.A_gpu);
    auto B_dev = copy_sparse_to_device(synth.B_gpu);

    PipelinePlan<T> plan;
    pipeline_setup<T>(plan, synth.A_gpu, synth.B_gpu, A_dev, B_dev,
                      synth.index_sets, synth.output_labels);

    if (plan.medium_vol_num_problems == 0) {
        printf("  N/A");
    } else {
        auto r = gpu_bench([&]() {
            stage_medium_gemm(plan, plan.stream_medium);
        }, warmup, repeat, plan.stream_medium);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  CUDA error: %s", cudaGetErrorString(err));
            cudaGetLastError();
        } else {
            double gflops = static_cast<double>(total_flops) / (r.median_ms * 1e6);
            printf("  %.3fms %6.0f GF/s", r.median_ms, gflops);
        }
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// Synthetic mode: sweep configurations
// ---------------------------------------------------------------------------

template <typename T>
void run_synthetic(const std::string& einsum_str, int num_descs,
                   int warmup, int repeat) {
    auto expr = parse_einsum(einsum_str);
    EinsumIndexSets idx;
    create_einsum_index_sets(expr, idx);

    printf("  free_A=%zu dims, free_B=%zu dims, contracted=%zu dims\n",
           idx.free_indices_A.size(), idx.free_indices_B.size(),
           idx.contracted_indices.size());

    // M sweep (N=9, K=32)
    {
        int N = 9, K = 32;
        printf("\n--- M sweep (N=%d, K=%d) ---\n", N, K);
        for (int M : {9, 12, 16, 20, 24, 28, 32}) {
            bench_synthetic_point<T>(einsum_str, M, N, K, num_descs,
                                     warmup, repeat);
        }
    }

    // N sweep (M=9, K=32)
    {
        int M = 9, K = 32;
        printf("\n--- N sweep (M=%d, K=%d) ---\n", M, K);
        for (int N : {9, 12, 16, 20, 24, 28, 32}) {
            bench_synthetic_point<T>(einsum_str, M, N, K, num_descs,
                                     warmup, repeat);
        }
    }

    // K sweep (M=32, N=32) — max spatial, K limited by VOL_THRESHOLD
    {
        int M = 32, N = 32;
        printf("\n--- K sweep (M=%d, N=%d) ---\n", M, N);
        for (int K : {1, 2, 4, 8, 16, 32}) {
            bench_synthetic_point<T>(einsum_str, M, N, K, num_descs,
                                     warmup, repeat);
        }
    }

    // K sweep (M=9, N=9) — small spatial, larger K budget
    {
        int M = 9, N = 9;
        printf("\n--- K sweep (M=%d, N=%d) ---\n", M, N);
        for (int K : {8, 16, 32, 64, 96, 113}) {
            bench_synthetic_point<T>(einsum_str, M, N, K, num_descs,
                                     warmup, repeat);
        }
    }

    // K sweep (M=16, N=16)
    {
        int M = 16, N = 16;
        printf("\n--- K sweep (M=%d, N=%d) ---\n", M, N);
        for (int K : {8, 16, 32, 64}) {
            bench_synthetic_point<T>(einsum_str, M, N, K, num_descs,
                                     warmup, repeat);
        }
    }

    // Square sweep (K=16)
    {
        int K = 16;
        printf("\n--- Square sweep (M=N, K=%d) ---\n", K);
        for (int S : {9, 12, 16, 20, 24, 28, 32}) {
            bench_synthetic_point<T>(einsum_str, S, S, K, num_descs,
                                     warmup, repeat);
        }
    }

    // Square sweep with larger K
    {
        int K = 64;
        printf("\n--- Square sweep (M=N, K=%d) ---\n", K);
        for (int S : {9, 12, 16}) {
            bench_synthetic_point<T>(einsum_str, S, S, K, num_descs,
                                     warmup, repeat);
        }
    }

    // Random sweep — constrained to volume-eligible region
    {
        printf("\n--- Random (50 points) ---\n");
        std::mt19937 rng(12345);
        int generated = 0;
        for (int attempts = 0; attempts < 500 && generated < 50; ++attempts) {
            int M = 9 + static_cast<int>(rng() % 24);   // [9, 32]
            int N = 9 + static_cast<int>(rng() % 24);   // [9, 32]
            int max_K = vol_threshold<T>() / (M + N);
            if (max_K < 1) continue;
            int K = 1 + static_cast<int>(rng() % max_K);
            bench_synthetic_point<T>(einsum_str, M, N, K, num_descs,
                                     warmup, repeat);
            ++generated;
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int warmup = 3, repeat = 5, num_descs = 256;
    bool use_f64 = false;
    bool synthetic = false;
    std::string filter;
    std::string einsum_str = "ij,jk->ik";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--num-descs") && i + 1 < argc) num_descs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dtype") && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "f64" || v == "float64") use_f64 = true;
        }
        else if (!strcmp(argv[i], "--filter") && i + 1 < argc) filter = argv[++i];
        else if (!strcmp(argv[i], "--tensor") && i + 1 < argc) filter = argv[++i];
        else if (!strcmp(argv[i], "--synthetic")) synthetic = true;
        else if (!strcmp(argv[i], "--einsum") && i + 1 < argc) einsum_str = argv[++i];
    }

    if (synthetic) {
        printf("Medium GEMM synthetic benchmark (einsum=%s, %d descs, warmup=%d, repeat=%d, %s)\n",
               einsum_str.c_str(), num_descs, warmup, repeat, use_f64 ? "f64" : "f32");
        try {
            if (use_f64)
                run_synthetic<double>(einsum_str, num_descs, warmup, repeat);
            else
                run_synthetic<float>(einsum_str, num_descs, warmup, repeat);
        } catch (const std::exception& e) {
            fprintf(stderr, "ERROR: %s\n", e.what());
            return 1;
        }
    } else {
        auto cases = discover_cases(filter);
        if (cases.empty()) {
            fprintf(stderr, "No tensor cases found (filter: '%s')\n", filter.c_str());
            return 1;
        }

        printf("Medium GEMM kernel benchmark (%zu cases, warmup=%d, repeat=%d)\n",
               cases.size(), warmup, repeat);

        for (const auto& tc : cases) {
            try {
                if (use_f64)
                    run_tensor_case<double>(tc, warmup, repeat);
                else
                    run_tensor_case<float>(tc, warmup, repeat);
            } catch (const std::exception& e) {
                printf("  ERROR: %s\n", e.what());
                cudaGetLastError();
            }
        }
    }

    return 0;
}
