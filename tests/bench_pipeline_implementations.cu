// Benchmark: compare CUTLASS pipeline vs cuTENSOR block-sparse
// on real cutensor_converted tensor data.
//
// Build: make bench_pipeline_implementations
// Run: ./bench_pipeline_implementations [--filter <substr>] [--warmup N] [--repeat N] [--no-verify]

#include "bs_parser.hpp"
#include "einsum.hpp"
#include "general_sparse/gpu_sparse_input.hpp"
#include "general_sparse/general_sparse_executor.cuh"
#include "general_sparse/stride_verifier.hpp"
#include "general_sparse/gemm_layout_check.hpp"
#include "general_sparse/cutensor_block_sparse.cuh"

#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <tuple>

using namespace bsptc;
namespace fs = std::filesystem;

static std::string g_filter;
static int g_warmup = 1;
static int g_repeat = 3;
static bool g_no_verify = false;
static bool g_no_cutensor = false;  // skip cuTENSOR block-sparse comparison entirely
static bool g_diag = false;
static bool g_f64 = false;
static int64_t g_streaming_cap_mb = 0;  // 0 = disabled; >0 = streaming transpose buffer cap in MB
static size_t g_initial_free_mem = 0;  // cached at startup, immune to cuTENSOR corruption
static bool g_cuda_context_dirty = false;  // set after cuTENSOR corrupts context

struct ContractionInfo {
    std::string einsum;
    int A_blocks, B_blocks;
    int A_rank, B_rank;
};

ContractionInfo parse_info_file(const std::string& path) {
    ContractionInfo info{};
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
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

// Run a single implementation and return median time in ms.
// impl_fn runs the contraction (and frees results).
// If capture_fn is provided, the last timed iteration calls capture_fn instead
// of impl_fn, allowing the caller to retain the result for later use.
double bench(int warmup, int repeat, std::function<void()> impl_fn,
             std::function<void()> capture_fn = nullptr) {
    // Warmup
    for (int i = 0; i < warmup; ++i) impl_fn();

    // Timed runs
    std::vector<double> times(repeat);
    for (int i = 0; i < repeat; ++i) {
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        if (capture_fn && i == repeat - 1)
            capture_fn();
        else
            impl_fn();
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    return times[repeat / 2]; // median
}

// Caller-visible allocations live outside prepare_ms.  Plan-owned metadata
// allocations remain part of prepare: they are implementation-private work
// required to make the next contraction call launch-ready.
struct ScopedDeviceBuffer {
    void* ptr = nullptr;

    ScopedDeviceBuffer() = default;
    explicit ScopedDeviceBuffer(int64_t bytes) {
        if (bytes > 0)
            check_cuda(cudaMalloc(&ptr, static_cast<size_t>(bytes)),
                       "prepare benchmark cudaMalloc");
    }
    ~ScopedDeviceBuffer() { if (ptr) cudaFree(ptr); }
    ScopedDeviceBuffer(const ScopedDeviceBuffer&) = delete;
    ScopedDeviceBuffer& operator=(const ScopedDeviceBuffer&) = delete;
};

double median_sample(std::vector<double>& samples) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

struct PrepareMeasurement {
    double host_prepare_ms = 0.0;
    double device_zero_ms = 0.0;
    double prepare_ms = 0.0;
};

bool cuda_context_is_usable() {
    // A failed cuTENSOR call can leave either a sticky error or a destroyed
    // context.  Clear the former and use a tiny allocation to distinguish it
    // from the latter before deciding whether the real benchmark may proceed.
    cudaGetLastError();
    cudaDeviceSynchronize();
    cudaGetLastError();
    void* probe = nullptr;
    auto rc = cudaMalloc(&probe, 256);
    if (rc != cudaSuccess) {
        cudaGetLastError();
        g_cuda_context_dirty = true;
        return false;
    }
    cudaFree(probe);
    return true;
}

// Resource-ready TIDES prepare: A/B host+device inputs, output sparsity, and
// caller-visible output/workspace allocations already exist.  Rebuild an
// uncached plan, prepare geometry, bind pointers, and finish metadata staging.
template <typename T>
PrepareMeasurement measure_tides_prepare(
    const GPUSparseInput<T>& A_gpu,
    const GPUSparseInput<T>& B_gpu,
    const DeviceSparseInput<T>& A_dev,
    const DeviceSparseInput<T>& B_dev,
    const EinsumIndexSets& index_sets,
    const std::string& output_labels,
    const DispatchConfig& dc,
    const OutputBlockMap& output_blocks)
{
    PipelineWorkspaceInfo ws_info;
    {
        PipelinePlan<T> size_plan;
        if (!pipeline_plan(size_plan, A_gpu, B_gpu, index_sets, output_labels,
                           dc, nullptr, &output_blocks) ||
            !pipeline_prepare(size_plan, A_gpu, B_gpu))
            throw std::runtime_error("TIDES prepare-size analysis failed");
        ws_info = pipeline_query_workspace(size_plan, A_gpu, B_gpu);
    }

    ScopedDeviceBuffer output(ws_info.output_bytes);
    ScopedDeviceBuffer workspace(ws_info.workspace_bytes);
    std::vector<double> host_samples;
    std::vector<double> zero_samples;
    std::vector<double> total_samples;
    host_samples.reserve(static_cast<size_t>(g_repeat));
    zero_samples.reserve(static_cast<size_t>(g_repeat));
    total_samples.reserve(static_cast<size_t>(g_repeat));

    for (int i = 0; i < g_warmup + g_repeat; ++i) {
        // Destruction (metadata free/repool) is deliberately after the timing
        // endpoint and belongs to memfree, not prepare.
        PipelinePlan<T> plan;
        check_cuda(cudaDeviceSynchronize(), "TIDES prepare pre-sync");
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!pipeline_plan(plan, A_gpu, B_gpu, index_sets, output_labels,
                           dc, nullptr, &output_blocks) ||
            !pipeline_prepare(plan, A_gpu, B_gpu) ||
            !pipeline_allocate(plan, A_gpu, B_gpu, A_dev, B_dev,
                               static_cast<T*>(output.ptr), workspace.ptr,
                               ws_info.workspace_bytes,
                               /*initialize_accumulators=*/false))
            throw std::runtime_error("TIDES resource-ready prepare failed");
        check_cuda(cudaDeviceSynchronize(), "TIDES host prepare post-sync");
        auto t1 = std::chrono::high_resolution_clock::now();

        const int64_t output_bytes = plan.accum_map.total_output_elements
                                   * static_cast<int64_t>(sizeof(T));
        const int64_t temp_bytes = plan.total_pair_output_size
                                 * static_cast<int64_t>(sizeof(T));
        if (plan.output_beta == 0 && output_bytes > 0)
            check_cuda(cudaMemset(plan.d_output, 0,
                                  static_cast<size_t>(output_bytes)),
                       "TIDES output initialization");
        if (temp_bytes > 0)
            check_cuda(cudaMemset(plan.d_temp_buffer, 0,
                                  static_cast<size_t>(temp_bytes)),
                       "TIDES temp initialization");
        check_cuda(cudaDeviceSynchronize(), "TIDES device zero post-sync");
        auto t2 = std::chrono::high_resolution_clock::now();

        if (i >= g_warmup) {
            host_samples.push_back(std::chrono::duration<double, std::milli>(
                                       t1 - t0).count());
            zero_samples.push_back(std::chrono::duration<double, std::milli>(
                                       t2 - t1).count());
            total_samples.push_back(std::chrono::duration<double, std::milli>(
                                        t2 - t0).count());
        }
    }
    return {median_sample(host_samples), median_sample(zero_samples),
            median_sample(total_samples)};
}

// Resource-ready cuTENSOR prepare.  A/B tensor descriptors are persistent,
// matching TIDES' persistent GPUSparseInput objects.  Output sparsity and the
// exact allocation sizes are analyzed before memalloc; prepare_ms then covers
// output/operation descriptors, pointer binding, preference, and plan creation.
template <typename T>
PrepareMeasurement measure_cutensor_prepare(
    const BlockSparseTensor& A_bst,
    const BlockSparseTensor& B_bst,
    const EinsumExpression& expr,
    const T* d_A,
    const T* d_B,
    const std::vector<std::vector<int64_t>>& output_coords)
{
    using detail_bs::CHandle;
    cutensorHandle_t handle = get_cached_cutensor_handle();
    auto a_desc = detail_bs::create_bs_descriptor<T>(handle, A_bst, d_A);
    auto b_desc = detail_bs::create_bs_descriptor<T>(handle, B_bst, d_B);

    if (output_coords.empty())
        throw std::invalid_argument("empty output has no cuTENSOR prepare interval");
    auto out_bst = build_output_bst_meta(A_bst, B_bst, expr, output_coords);
    const uint64_t out_nnz = static_cast<uint64_t>(output_coords.size());

    std::vector<int64_t> out_offsets;
    auto [out_volumes, out_total_elements] =
        compute_block_volumes_and_offsets(out_bst, output_coords, out_offsets);

    auto create_output_desc = [&](cutensorBlockSparseTensorDescriptor_t* desc) {
        // These vectors are the per-contraction descriptor packing that the
        // cuTENSOR API requires.  Build coordinates directly from the common
        // output list; build_output_bst_meta itself is untimed harness work.
        std::vector<int64_t> out_extents;
        std::vector<uint32_t> out_sections;
        detail_bs::build_extent_arrays(out_bst, out_extents, out_sections);
        std::vector<int32_t> out_flat_coords;
        out_flat_coords.reserve(output_coords.size()
                                * static_cast<size_t>(out_bst.meta.ndim));
        for (const auto& coord : output_coords)
            for (int64_t c : coord)
                out_flat_coords.push_back(static_cast<int32_t>(c));
        detail_bs::check_cutensor(
            cutensorCreateBlockSparseTensorDescriptor(
                handle, desc, static_cast<uint32_t>(out_bst.meta.ndim), out_nnz,
                out_sections.data(), out_extents.data(), out_flat_coords.data(),
                nullptr, to_cutensor_type(out_bst.meta.dtype)),
            "cutensorCreateBlockSparseTensorDescriptor (prepare benchmark)");
    };

    auto make_modes = [&]() {
        std::vector<int32_t> mA, mB, mO;
        for (char c : A_bst.meta.indices) mA.push_back(static_cast<int32_t>(c));
        for (char c : B_bst.meta.indices) mB.push_back(static_cast<int32_t>(c));
        for (char c : expr.output_labels) mO.push_back(static_cast<int32_t>(c));
        return std::make_tuple(std::move(mA), std::move(mB), std::move(mO));
    };

    // Analyze exact workspace size before caller-visible allocation.  The
    // timed prepare recreates the uncached operation and plan with this size.
    uint64_t ws_size = 0;
    {
        CHandle<cutensorBlockSparseTensorDescriptor_t,
                cutensorDestroyBlockSparseTensorDescriptor> out_desc;
        CHandle<cutensorOperationDescriptor_t,
                cutensorDestroyOperationDescriptor> op_desc;
        CHandle<cutensorPlanPreference_t,
                cutensorDestroyPlanPreference> pref;
        create_output_desc(&out_desc.h);
        auto [mA, mB, mO] = make_modes();
        detail_bs::check_cutensor(
            cutensorCreateBlockSparseContraction(
                handle, &op_desc.h,
                a_desc.desc, mA.data(), CUTENSOR_OP_IDENTITY,
                b_desc.desc, mB.data(), CUTENSOR_OP_IDENTITY,
                out_desc.h, mO.data(), CUTENSOR_OP_IDENTITY,
                out_desc.h, mO.data(), to_cutensor_compute(A_bst.meta.dtype)),
            "cutensorCreateBlockSparseContraction (prepare-size analysis)");
        detail_bs::check_cutensor(
            cutensorCreatePlanPreference(handle, &pref.h, CUTENSOR_ALGO_DEFAULT,
                                         CUTENSOR_JIT_MODE_NONE),
            "cutensorCreatePlanPreference (prepare-size analysis)");
        detail_bs::check_cutensor(
            cutensorEstimateWorkspaceSize(handle, op_desc.h, pref.h,
                                          CUTENSOR_WORKSPACE_DEFAULT, &ws_size),
            "cutensorEstimateWorkspaceSize (prepare-size analysis)");
    }

    const int64_t output_bytes =
        out_total_elements * static_cast<int64_t>(sizeof(T));

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
               "cuTENSOR prepare cudaMemGetInfo");
    const uint64_t output_bytes_u = static_cast<uint64_t>(output_bytes);
    if (ws_size > std::numeric_limits<uint64_t>::max() - output_bytes_u)
        throw std::runtime_error("cuTENSOR prepare allocation size overflow");
    const uint64_t required_bytes = output_bytes_u + ws_size;
    if (required_bytes > static_cast<uint64_t>(free_bytes)) {
        auto mib = [](uint64_t bytes) {
            return static_cast<double>(bytes) / (1024.0 * 1024.0);
        };
        std::ostringstream msg;
        msg << "cuTENSOR prepare OOM: need "
            << mib(required_bytes) << " MB (output="
            << mib(output_bytes_u) << " + workspace="
            << mib(ws_size) << "), free="
            << mib(static_cast<uint64_t>(free_bytes)) << " MB";
        throw std::runtime_error(msg.str());
    }

    ScopedDeviceBuffer output(output_bytes);
    ScopedDeviceBuffer workspace(static_cast<int64_t>(ws_size));
    std::vector<double> host_samples;
    std::vector<double> zero_samples;
    std::vector<double> total_samples;
    host_samples.reserve(static_cast<size_t>(g_repeat));
    zero_samples.reserve(static_cast<size_t>(g_repeat));
    total_samples.reserve(static_cast<size_t>(g_repeat));

    for (int i = 0; i < g_warmup + g_repeat; ++i) {
        CHandle<cutensorBlockSparseTensorDescriptor_t,
                cutensorDestroyBlockSparseTensorDescriptor> out_desc;
        CHandle<cutensorOperationDescriptor_t,
                cutensorDestroyOperationDescriptor> op_desc;
        CHandle<cutensorPlanPreference_t,
                cutensorDestroyPlanPreference> pref;
        CHandle<cutensorPlan_t, cutensorDestroyPlan> plan;
        CHandle<cudaStream_t, cudaStreamDestroy> stream;
        CHandle<cudaEvent_t, cudaEventDestroy> ev0, ev1;
        std::vector<const void*> c_ptrs;
        std::vector<void*> d_ptrs;

        check_cuda(cudaDeviceSynchronize(), "cuTENSOR prepare pre-sync");
        auto t0 = std::chrono::high_resolution_clock::now();
        create_output_desc(&out_desc.h);

        c_ptrs.resize(out_nnz);
        d_ptrs.resize(out_nnz);
        int64_t ptr_offset = 0;
        for (uint64_t b = 0; b < out_nnz; ++b) {
            T* ptr = static_cast<T*>(output.ptr) + ptr_offset;
            c_ptrs[b] = static_cast<const void*>(ptr);
            d_ptrs[b] = static_cast<void*>(ptr);
            ptr_offset += out_volumes[b];
        }
#if defined(__GNUC__)
        // The real API consumes these at launch.  The measurement ends before
        // launch, so make their stores observable to the optimizing compiler.
        asm volatile("" : : "r"(c_ptrs.data()), "r"(d_ptrs.data()) : "memory");
#endif

        auto [mA, mB, mO] = make_modes();
        detail_bs::check_cutensor(
            cutensorCreateBlockSparseContraction(
                handle, &op_desc.h,
                a_desc.desc, mA.data(), CUTENSOR_OP_IDENTITY,
                b_desc.desc, mB.data(), CUTENSOR_OP_IDENTITY,
                out_desc.h, mO.data(), CUTENSOR_OP_IDENTITY,
                out_desc.h, mO.data(), to_cutensor_compute(A_bst.meta.dtype)),
            "cutensorCreateBlockSparseContraction (prepare benchmark)");
        detail_bs::check_cutensor(
            cutensorCreatePlanPreference(handle, &pref.h, CUTENSOR_ALGO_DEFAULT,
                                         CUTENSOR_JIT_MODE_NONE),
            "cutensorCreatePlanPreference (prepare benchmark)");
        detail_bs::check_cutensor(
            cutensorCreatePlan(handle, &plan.h, op_desc.h, pref.h, ws_size),
            "cutensorCreatePlan (prepare benchmark)");
        check_cuda(cudaStreamCreate(&stream.h),
                   "cuTENSOR prepare stream creation");
        check_cuda(cudaEventCreate(&ev0.h),
                   "cuTENSOR prepare start-event creation");
        check_cuda(cudaEventCreate(&ev1.h),
                   "cuTENSOR prepare stop-event creation");
        check_cuda(cudaDeviceSynchronize(), "cuTENSOR prepare post-sync");
        auto t1 = std::chrono::high_resolution_clock::now();

        if (output_bytes > 0)
            check_cuda(cudaMemset(output.ptr, 0,
                                  static_cast<size_t>(output_bytes)),
                       "cuTENSOR output initialization");
        check_cuda(cudaDeviceSynchronize(), "cuTENSOR device zero post-sync");
        auto t2 = std::chrono::high_resolution_clock::now();

        if (i >= g_warmup) {
            host_samples.push_back(std::chrono::duration<double, std::milli>(
                                       t1 - t0).count());
            zero_samples.push_back(std::chrono::duration<double, std::milli>(
                                       t2 - t1).count());
            total_samples.push_back(std::chrono::duration<double, std::milli>(
                                        t2 - t0).count());
        }
    }
    return {median_sample(host_samples), median_sample(zero_samples),
            median_sample(total_samples)};
}

// Print a pipeline timing breakdown on one line
void print_breakdown(const char* label, const PipelineTimingBreakdown& tb) {
    std::cout << std::fixed << std::setprecision(2)
              << "  [" << label << "]";
    if (tb.transpose_ms > 0.005) std::cout << " transpose=" << tb.transpose_ms;
    std::cout << " contract=" << tb.contraction_ms
              << " accumulate=" << tb.accumulation_ms;
    if (tb.output_permute_ms > 0.005) std::cout << " out_permute=" << tb.output_permute_ms;
    std::cout << " total=" << tb.total_ms
              << " ms\n";
}

template <typename T>
void run_bench(const std::string& name, const std::string& dir_path, const ContractionInfo& info) {
    std::cout << name << "  einsum=" << info.einsum
              << "  A_blk=" << info.A_blocks << "  B_blk=" << info.B_blocks << "\n";

    // cuTENSOR errors can permanently corrupt the CUDA context
    if (g_cuda_context_dirty) {
        std::cout << "  SKIPPED (CUDA context corrupted by previous cuTENSOR error)\n\n"
                  << std::flush;
        return;
    }

    try {
        std::string fileA = dir_path + "/" + name + "_A.bs";
        std::string fileB = dir_path + "/" + name + "_B.bs";

        auto t_init = std::chrono::high_resolution_clock::now();

        DataType dtype = std::is_same<T, double>::value ? DataType::FLOAT64 : DataType::FLOAT32;
        auto tensors = parse_tensor_files({fileA, fileB}, dtype);
        BlockSparseTensor& A = tensors[0];
        BlockSparseTensor& B = tensors[1];

        auto expr = parse_einsum(info.einsum);
        std::vector<std::vector<int64_t>> shapes = {
            block_utils::compute_full_shape(A), block_utils::compute_full_shape(B)
        };
        validate_einsum(expr, shapes);

        EinsumIndexSets index_sets;
        create_einsum_index_sets(expr, index_sets);

        auto A_gpu = to_gpu_input<T>(A);
        auto B_gpu = to_gpu_input<T>(B);

        // Quick pre-check: skip if inputs alone won't fit in GPU memory
        int64_t A_bytes = A_gpu.get_values_count() * static_cast<int64_t>(sizeof(T));
        int64_t B_bytes = B_gpu.get_values_count() * static_cast<int64_t>(sizeof(T));
        int64_t input_bytes = A_bytes + B_bytes;
        double mem_gate = 0.7;
        if (const char* gate_env = std::getenv("BENCH_MEM_GATE")) {
            double v = std::atof(gate_env);
            if (v > 0.0 && v <= 1.0) mem_gate = v;
        }
        if (input_bytes > static_cast<int64_t>(static_cast<double>(g_initial_free_mem) * mem_gate)) {
            double est_gb = static_cast<double>(input_bytes) / (1024.0*1024.0*1024.0);
            double free_gb = static_cast<double>(g_initial_free_mem) / (1024.0*1024.0*1024.0);
            std::cout << "  SKIPPED (inputs " << std::fixed << std::setprecision(1)
                      << est_gb << " GB > " << static_cast<int>(mem_gate * 100)
                      << "% of available " << free_gb
                      << " GB; raise with BENCH_MEM_GATE)\n\n" << std::flush;
            return;
        }

        // Build dispatch config
        DispatchConfig dc;
        if (g_streaming_cap_mb > 0) {
            dc.transpose_buf_cap = (g_streaming_cap_mb * 1024 * 1024) / sizeof(T);
        }

        // Run pipeline_plan to get exact workspace sizes and layout info.
        // This replaces the old heuristic OOM estimate with precise numbers.
        PipelinePlan<T> probe_plan;
        pipeline_plan(probe_plan, A_gpu, B_gpu,
                      index_sets, expr.output_labels, dc);
        auto ws_info = pipeline_query_workspace(probe_plan, A_gpu, B_gpu);

        int64_t estimated_peak = input_bytes + ws_info.output_bytes + ws_info.workspace_bytes;

        // Save layout info from probe plan for reporting
        GemmLayoutPlan layout_plan = probe_plan.layout_plan;
        bool needs_transpose = probe_plan.layout_plan.plan_A.transpose ||
                               probe_plan.layout_plan.plan_B.transpose;
        ContractionConfig config = probe_plan.config;

        // Resource-ready prepare uses the same precomputed output sparsity for
        // both implementations.  Discovery belongs to the preceding analyze
        // phase because exact output allocation sizes depend on it.
        OutputBlockMap prepare_output_blocks = compute_output_blocks(
            probe_plan.pairs_host.data(), probe_plan.num_pairs,
            A_gpu, B_gpu, config);
        auto prepare_output_coords = prepare_output_blocks.to_output_order(
            config.internal_to_output);

        // Check peak GPU memory against available.
        // Use cached initial free memory (cuTENSOR errors can corrupt cudaMemGetInfo).
        if (estimated_peak > static_cast<int64_t>(static_cast<double>(g_initial_free_mem) * 0.9)) {
            double est_gb = static_cast<double>(estimated_peak) / (1024.0*1024.0*1024.0);
            double free_gb = static_cast<double>(g_initial_free_mem) / (1024.0*1024.0*1024.0);
            std::cout << "  SKIPPED (estimated " << std::fixed << std::setprecision(1)
                      << est_gb << " GB > available " << free_gb << " GB"
                      << "  [in=" << std::setprecision(1) << double(input_bytes)/(1024*1024*1024.0)
                      << " out=" << double(ws_info.output_bytes)/(1024*1024*1024.0)
                      << " ws=" << double(ws_info.workspace_bytes)/(1024*1024*1024.0)
                      << " t=" << (needs_transpose ? "Y" : "N") << "])\n\n" << std::flush;
            return;
        }

        double init_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_init).count();

        // Tier diagnostic: reuse preprocessing from probe plan
        if (g_diag && probe_plan.num_pairs > 0) {
            print_tier_diagnostic(A_gpu, B_gpu, probe_plan.pairs_host, config);
        }

        // 1. CUTLASS pipeline — upload inputs once, shared across iterations
        cudaEvent_t h2d_start, h2d_stop;
        cudaEventCreate(&h2d_start); cudaEventCreate(&h2d_stop);
        cudaEventRecord(h2d_start);
        DeviceSparseInput<T> A_dev = copy_sparse_to_device(A_gpu);
        DeviceSparseInput<T> B_dev = copy_sparse_to_device(B_gpu);
        cudaEventRecord(h2d_stop);
        cudaEventSynchronize(h2d_stop);
        float h2d_ms = 0;
        cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop);
        cudaEventDestroy(h2d_start); cudaEventDestroy(h2d_stop);
        double h2d_mb = static_cast<double>(
            A_gpu.get_values_count() + B_gpu.get_values_count()) * sizeof(T) / 1e6;

        PrepareMeasurement tides_prepare;
        std::string tides_prepare_status = "not_run";
        if (prepare_output_coords.empty()) {
            tides_prepare_status = "empty";
        } else {
            try {
                tides_prepare = measure_tides_prepare(
                    A_gpu, B_gpu, A_dev, B_dev, index_sets,
                    expr.output_labels, dc, prepare_output_blocks);
                tides_prepare_status = "ok";
            } catch (const std::exception& e) {
                tides_prepare_status = "error";
                std::cout << "  TIDES prepare measurement error: "
                          << e.what() << "\n";
                if (!cuda_context_is_usable())
                    throw std::runtime_error(
                        "CUDA context corrupted during TIDES prepare measurement");
            }
        }

        GeneralSparseResult<T> captured_pipeline_result;
        bool have_pipeline_result = false;

        PipelineTimingBreakdown cutlass_breakdown;
        // Full wall-clock per invocation: setup + execute + extract + free
        // (inputs pre-uploaded, excluded). Median over repeats.
        double pipeline_wall_ms = bench(g_warmup, g_repeat,
            [&]() {
                auto r = execute_general_sparse_pipeline(
                    A_gpu, B_gpu, A_dev, B_dev,
                    index_sets, expr.output_labels, dc);
            },
            [&]() {
                captured_pipeline_result = execute_general_sparse_pipeline(
                    A_gpu, B_gpu, A_dev, B_dev,
                    index_sets, expr.output_labels, dc,
                    &cutlass_breakdown);
                have_pipeline_result = true;
            });

        // Move pipeline result to host to free GPU memory before cuTENSOR.
        // A_dev/B_dev stay alive — cuTENSOR reuses the same device buffers.
        std::vector<T> pipeline_host_output;
        if (have_pipeline_result && captured_pipeline_result.d_output &&
            captured_pipeline_result.total_output_elements > 0) {
            pipeline_host_output.resize(captured_pipeline_result.total_output_elements);
            cudaMemcpy(pipeline_host_output.data(), captured_pipeline_result.d_output,
                       captured_pipeline_result.total_output_elements * sizeof(T),
                       cudaMemcpyDeviceToHost);
            cudaFree(captured_pipeline_result.d_output);
            captured_pipeline_result.d_output = nullptr;
            captured_pipeline_result.owns_output = false;
        }

        // 2. cuTENSOR block-sparse — reuse device inputs from pipeline
        CutensorBlockSparseResult<T> bs_last_result;
        bool have_bs_result = false;
        double bs_wall_ms = 0;
        PrepareMeasurement bs_prepare;
        std::string bs_prepare_status = g_no_cutensor ? "skipped" : "not_run";
        if (!g_no_cutensor) {
            const T* d_A = A_dev.d_values;
            const T* d_B = B_dev.d_values;
            if (prepare_output_coords.empty()) {
                bs_prepare_status = "empty";
            } else {
                try {
                    bs_prepare = measure_cutensor_prepare<T>(
                        A, B, expr, d_A, d_B, prepare_output_coords);
                    bs_prepare_status = "ok";
                } catch (const std::exception& e) {
                    bs_prepare_status = "error";
                    std::cout << "  cuTENSOR prepare measurement error: "
                              << e.what() << "\n";
                    cuda_context_is_usable();
                }
            }

            // Measurement failures are diagnostics only.  If the context is
            // still usable, independently attempt the real contraction and
            // verification path so one bad sample cannot erase kernel data.
            if (!g_cuda_context_dirty) {
                try {
                    bs_wall_ms = bench(g_warmup, g_repeat,
                        [&]() {
                            auto r = run_cutensor_block_sparse<T>(
                                A, B, expr, d_A, d_B,
                                &prepare_output_coords);
                        },
                        [&]() {
                            bs_last_result = run_cutensor_block_sparse<T>(
                                A, B, expr, d_A, d_B,
                                &prepare_output_coords);
                        });
                    have_bs_result = true;
                } catch (const std::exception& e) {
                    std::cout << "  cuTENSOR contraction error: "
                              << e.what() << "\n";
                    cuda_context_is_usable();
                }
            }
        }

        // Free device inputs — no longer needed after pipeline + cuTENSOR
        A_dev = DeviceSparseInput<T>();
        B_dev = DeviceSparseInput<T>();

        // 3. Correctness check (pipeline vs cuTENSOR block-sparse)
        auto t_verify = std::chrono::high_resolution_clock::now();
        bool correct = false;
        double max_rel = 0;
        bool have_correctness = false;

        if (!g_no_verify && have_pipeline_result && have_bs_result) {
            // Re-upload pipeline output to GPU for verification
            if (!pipeline_host_output.empty() && !captured_pipeline_result.d_output) {
                T* d_tmp = nullptr;
                cudaMalloc(&d_tmp, pipeline_host_output.size() * sizeof(T));
                cudaMemcpy(d_tmp, pipeline_host_output.data(),
                           pipeline_host_output.size() * sizeof(T), cudaMemcpyHostToDevice);
                captured_pipeline_result.d_output = d_tmp;
                captured_pipeline_result.owns_output = true;
            }
            auto vr = verify_against_block_sparse(
                captured_pipeline_result, A_gpu, B_gpu, config, bs_last_result);
            correct = vr.passed;
            max_rel = vr.max_rel;
            have_correctness = true;
        }
        double verify_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_verify).count();

        double cutlass_total = cutlass_breakdown.total_ms;

        // Report: summary line
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  layout=" << gemm_mode_to_string(layout_plan.mode_after_transpose);
        if (needs_transpose) {
            std::cout << " transpose=";
            if (layout_plan.transpose_A) std::cout << "A";
            if (layout_plan.transpose_B) std::cout << (layout_plan.transpose_A ? "+B" : "B");
        }
        if (have_correctness)
            std::cout << "  correct=" << (correct ? "yes" : "NO")
                      << "  max_rel=" << std::scientific << std::setprecision(2) << max_rel;
        else
            std::cout << "  correct=N/A";
        std::cout << "\n"
                  << std::fixed << std::setprecision(3)
                  << "  cutlass=" << std::setw(10) << cutlass_total << " ms";
        if (have_bs_result)
            std::cout << "  bs_cutensor=" << std::setw(10) << bs_last_result.kernel_ms << " ms";
        else if (g_no_cutensor)
            std::cout << "  bs_cutensor=   skipped";
        else
            std::cout << "  bs_cutensor=     ERROR";
        std::cout << "\n";

        // Report: timing breakdowns
        print_breakdown("cutlass", cutlass_breakdown);
        {
            const auto& tb = cutlass_breakdown;
            int64_t total_flops = tb.large_flops + tb.medium_flops + tb.small_flops;
            double total_gflop = static_cast<double>(total_flops) / 1e9;

            if (tb.large_gemm_problems > 0) {
                std::cout << std::fixed << std::setprecision(2)
                          << "  [large] problems=" << tb.large_gemm_problems
                          << "  flops=" << (static_cast<double>(tb.large_flops) / 1e9) << " GFLOP"
                          << "  output=" << tb.large_output_elems << " elems";
                if (!tb.large_tile_info.empty())
                    std::cout << "  tile=" << tb.large_tile_info;
                std::cout << "\n";
                if (tb.streaming_enabled) {
                    std::cout << "  [streaming] " << tb.streaming_total_batches << " batches"
                              << "  buf=" << (tb.streaming_buf_cap * sizeof(T) / (1024*1024)) << " MB x2"
                              << "  precomp=" << (tb.streaming_precomp_bytes / 1024) << " KB";
                    for (int qi = 0; qi < bsptc::NUM_SHAPE_QUEUES; ++qi) {
                        const auto& qi_info = tb.streaming_queues[qi];
                        if (qi_info.batches == 0) continue;
                        std::cout << "  " << bsptc::shape_queue_name(static_cast<bsptc::ShapeQueue>(qi))
                                  << "=" << qi_info.problems << "p/" << qi_info.batches << "b"
                                  << "/" << qi_info.tile_name;
                    }
                    std::cout << "\n";
                } else if (g_streaming_cap_mb > 0) {
                    std::cout << "  [streaming OFF] reason: no transpose needed\n";
                }
            }
            if (tb.large_gemm_problems == 0 && g_streaming_cap_mb > 0) {
                std::cout << "  [streaming OFF] reason: no large blocks\n";
            }
            if (tb.medium_descs > 0) {
                std::cout << std::fixed << std::setprecision(2)
                          << "  [medium] descs=" << tb.medium_descs
                          << "  flops=" << (static_cast<double>(tb.medium_flops) / 1e9) << " GFLOP"
                          << "  output=" << tb.medium_output_elems << " elems\n";
            }
            if (tb.small_descs > 0) {
                std::cout << "  [small] descs=" << tb.small_descs
                          << " shapes=" << tb.small_shapes
                          << " warps=" << tb.small_warps
                          << std::fixed << std::setprecision(2)
                          << "  flops=" << (static_cast<double>(tb.small_flops) / 1e9) << " GFLOP"
                          << "  output=" << tb.small_output_elems << " elems";
                if (tb.small_kernel_kind)
                    std::cout << "  kernel=" << tb.small_kernel_kind;
                std::cout << "\n";
            }
            if (tb.transpose_A_kind || tb.transpose_B_kind) {
                std::cout << "  [transpose] A=" << (tb.transpose_A_kind ? tb.transpose_A_kind : "none")
                          << "  B=" << (tb.transpose_B_kind ? tb.transpose_B_kind : "none") << "\n";
                if (!tb.transpose_A_detail.empty())
                    std::cout << "  [transpose_A] " << tb.transpose_A_detail << "\n";
                if (!tb.transpose_B_detail.empty())
                    std::cout << "  [transpose_B] " << tb.transpose_B_detail << "\n";
            }
            if (total_flops > 0) {
                std::cout << std::fixed << std::setprecision(2)
                          << "  [e2e] flops=" << total_gflop << " GFLOP"
                          << "  GFLOP/s=" << std::setprecision(1)
                          << (static_cast<double>(total_flops) / (tb.total_ms * 1e6))
                          << "\n";
            }
            if (tb.peak_gpu_bytes > 0) {
                std::cout << std::fixed << std::setprecision(1)
                          << "  [memory] peak=" << (static_cast<double>(tb.peak_gpu_bytes) / 1e6) << " MB\n";
            }
        }
        if (have_bs_result) {
            std::cout << std::fixed << std::setprecision(2)
                      << "  [bs_cutensor] kernel=" << bs_last_result.kernel_ms << " ms";
            if (bs_last_result.peak_gpu_bytes > 0)
                std::cout << std::setprecision(1)
                          << "  peak=" << (static_cast<double>(bs_last_result.peak_gpu_bytes) / 1e6) << " MB";
            std::cout << "\n";
        }

        // Matched resource-ready preparation: caller-visible output/workspace
        // allocations and persistent input metadata/descriptors are excluded.
        std::cout << std::fixed << std::setprecision(3)
                  << "  [prepare] tides_status=" << tides_prepare_status;
        if (tides_prepare_status == "ok") {
            std::cout << " tides_host_prepare_ms=" << tides_prepare.host_prepare_ms
                      << " tides_device_zero_ms=" << tides_prepare.device_zero_ms
                      << " tides_prepare_ms=" << tides_prepare.prepare_ms;
        }
        std::cout << " cutensor_status=" << bs_prepare_status;
        if (bs_prepare_status == "ok") {
            std::cout << " cutensor_host_prepare_ms=" << bs_prepare.host_prepare_ms
                      << " cutensor_device_zero_ms=" << bs_prepare.device_zero_ms
                      << " cutensor_prepare_ms=" << bs_prepare.prepare_ms;
        }
        std::cout << "\n";

        // TIDES-only full setup diagnostics.  This includes allocation and is
        // intentionally not compared against cuTENSOR prepare_ms.
        double cutlass_setup_ms = cutlass_breakdown.plan_ms
            + cutlass_breakdown.prepare_ms + cutlass_breakdown.allocate_ms;
        std::cout << std::fixed << std::setprecision(2)
                  << "  [setup_detail]"
                  << " tides_full_setup=" << cutlass_setup_ms
                  << " plan=" << cutlass_breakdown.plan_ms
                  << " prepare=" << cutlass_breakdown.prepare_ms
                  << " allocate=" << cutlass_breakdown.allocate_ms
                  << " preproc=" << cutlass_breakdown.preprocess_ms
                  << " accum_map=" << cutlass_breakdown.accum_map_ms
                  << " bind_upload=" << cutlass_breakdown.bind_upload_ms
                  << " bind_patch=" << cutlass_breakdown.bind_patch_ms
                  << " bind_stream=" << cutlass_breakdown.bind_stream_ms;
        std::cout << " h2d=" << h2d_ms
                  << " h2d_MB=" << std::setprecision(1) << h2d_mb;
        std::cout << std::setprecision(2) << " init=" << init_ms;
        if (have_correctness) std::cout << " verify=" << verify_ms;
        std::cout << " ms\n";

        // End-to-end wall clock (setup + execute + extract + free,
        // pre-uploaded inputs).  Diagnostic only, NOT a matched comparison:
        // cuTENSOR is handed the precomputed output coordinates while the
        // pipeline rediscovers output sparsity itself.  Use the [prepare]
        // line for setup comparisons.
        std::cout << std::fixed << std::setprecision(2)
                  << "  [wall] pipeline=" << pipeline_wall_ms;
        if (have_bs_result) std::cout << " bs_cutensor=" << bs_wall_ms;
        std::cout << " ms\n";

        std::cout << "\n" << std::flush;
    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << "\n\n" << std::flush;
        // Clear sticky CUDA errors to prevent cascade failures
        cudaGetLastError();
        cudaDeviceSynchronize();
        cudaGetLastError();
    }
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--filter" || arg == "-f") && i + 1 < argc) g_filter = argv[++i];
        else if (arg == "--warmup" && i + 1 < argc) g_warmup = std::stoi(argv[++i]);
        else if (arg == "--repeat" && i + 1 < argc) g_repeat = std::stoi(argv[++i]);
        else if (arg == "--no-verify") g_no_verify = true;
        else if (arg == "--no-cutensor") { g_no_cutensor = true; g_no_verify = true; }
        else if (arg == "--diag") g_diag = true;
        else if (arg == "--streaming" && i + 1 < argc) g_streaming_cap_mb = std::stoll(argv[++i]);
        else if (arg == "--dtype" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "f64" || v == "float64") g_f64 = true;
        }
    }

    std::cout << "========================================\n";
    std::cout << "Pipeline Implementation Benchmark\n";
    std::cout << "========================================\n\n";

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Cache initial free memory — cuTENSOR errors can corrupt cudaMemGetInfo
    size_t total_mem = 0;
    cudaMemGetInfo(&g_initial_free_mem, &total_mem);
    std::cout << "GPU: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")\n";
    std::cout << "Warmup: " << g_warmup << "  Repeat: " << g_repeat;
    if (g_no_verify) std::cout << "  Verify: disabled";
    if (g_f64) std::cout << "  Dtype: f64";
    if (g_streaming_cap_mb > 0) std::cout << "  Streaming: " << g_streaming_cap_mb << " MB";
    std::cout << "\n";
    std::cout << "\n";
    std::cout << std::flush;

    std::string data_root = "../generated_tensors/cutensor_converted";

    std::vector<std::string> test_dirs;
    for (const auto& entry : fs::directory_iterator(data_root)) {
        if (!entry.is_directory()) continue;
        std::string dirname = entry.path().filename().string();
        if (dirname.find("_E") == std::string::npos && dirname.find("_S") == std::string::npos) continue;
        if (!g_filter.empty() && dirname.find(g_filter) == std::string::npos) continue;
        test_dirs.push_back(dirname);
    }
    std::sort(test_dirs.begin(), test_dirs.end());

    std::cout << "Test cases: " << test_dirs.size();
    if (!g_filter.empty()) std::cout << " (filter: " << g_filter << ")";
    std::cout << "\n\n" << std::flush;

    for (const auto& dirname : test_dirs) {
        std::string dir_path = data_root + "/" + dirname;
        std::string info_path = dir_path + "/" + dirname + "_info.txt";

        ContractionInfo info;
        try { info = parse_info_file(info_path); }
        catch (...) { std::cout << dirname << "  SKIPPED (no info file)\n\n" << std::flush; continue; }

        if (g_f64)
            run_bench<double>(dirname, dir_path, info);
        else
            run_bench<float>(dirname, dir_path, info);
    }

    std::cout << "========================================\n";
    std::cout << "Done.\n";
    return 0;
}
