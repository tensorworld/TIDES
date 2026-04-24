// Standalone hypertile transpose tester and statistics reporter.
// Loads .bs tensor files, computes layout plan, runs hypertile transpose,
// verifies correctness against a naive reference, and reports detailed
// tiling statistics for kernel tuning.
//
// Two modes:
//   Einsum mode:  --a <A.bs> --b <B.bs> --einsum "ij,jk->ik"
//   Perm mode:    --tensor <T.bs> --perm "2,0,1"
//
// Options:
//   --dtype f32|f64        Data type (default: f32)
//   --warmup N             Warmup iterations (default: 3)
//   --repeat N             Timed iterations (default: 5)
//   --verbose              Print per-block task details
//   --no-verify            Skip correctness verification
//   --stats-only           Only print statistics, no kernel launch
//   --tile-vol N           Override hypertile tile volume (default: 4096)

#include "bs_parser.hpp"
#include "einsum.hpp"
#include "general_sparse/gpu_sparse_input.hpp"
#include "general_sparse/gpu_sparse_device.cuh"
#include "general_sparse/gemm_layout_check.hpp"
#include "general_sparse/transpose_kernel.cuh"
#include "general_sparse/kernels/transpose/nd_hypertile.cuh"
#include "general_sparse/pipeline.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <filesystem>
#include <fstream>
#include <chrono>

namespace fs = std::filesystem;

using namespace bsptc;

enum class TilerMode { Greedy, Search, Compare };

struct Options {
    // Einsum mode
    std::string tensor_a;
    std::string tensor_b;
    std::string einsum_str;
    // Perm mode
    std::string tensor_path;
    std::string perm_str;
    // Shape mode (synthetic single block)
    std::string shape_str;
    // Scan mode
    std::string scan_dir;

    DataType dtype = DataType::FLOAT32;
    int warmup = 3;
    int repeat = 5;
    bool verbose = false;
    bool verify = true;
    bool stats_only = false;
    int tile_vol = HT_TILE_VOL;  // match pipeline default
    TilerMode tiler = TilerMode::Greedy;

    bool is_shape_mode() const { return !shape_str.empty() && !perm_str.empty(); }
    bool is_perm_mode() const { return !tensor_path.empty() && !perm_str.empty(); }
    bool is_einsum_mode() const { return !tensor_a.empty() && !tensor_b.empty() && !einsum_str.empty(); }
    bool is_scan_mode() const { return !scan_dir.empty(); }
};

// Parse "37x37x37x37x35" into vector of dims.
std::vector<int32_t> parse_shape_string(const std::string& s) {
    std::vector<int32_t> dims;
    std::string tok;
    for (char c : s) {
        if (c == 'x' || c == 'X') {
            dims.push_back(std::atoi(tok.c_str()));
            tok.clear();
        } else {
            tok += c;
        }
    }
    if (!tok.empty()) dims.push_back(std::atoi(tok.c_str()));
    return dims;
}

// Parse comma-separated perm string like "2,0,1" into vector.
std::vector<int32_t> parse_perm_string(const std::string& s) {
    std::vector<int32_t> perm;
    std::string tok;
    for (char c : s) {
        if (c == ',') {
            perm.push_back(std::atoi(tok.c_str()));
            tok.clear();
        } else {
            tok += c;
        }
    }
    if (!tok.empty()) perm.push_back(std::atoi(tok.c_str()));
    return perm;
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--a" && i + 1 < argc)       opts.tensor_a = argv[++i];
        else if (arg == "--b" && i + 1 < argc)  opts.tensor_b = argv[++i];
        else if (arg == "--einsum" && i + 1 < argc) opts.einsum_str = argv[++i];
        else if (arg == "--tensor" && i + 1 < argc) opts.tensor_path = argv[++i];
        else if (arg == "--shape" && i + 1 < argc)  opts.shape_str = argv[++i];
        else if (arg == "--perm" && i + 1 < argc)   opts.perm_str = argv[++i];
        else if (arg == "--dtype" && i + 1 < argc) {
            std::string d = argv[++i];
            opts.dtype = (d == "f64" || d == "double") ? DataType::FLOAT64 : DataType::FLOAT32;
        }
        else if (arg == "--warmup" && i + 1 < argc) opts.warmup = std::atoi(argv[++i]);
        else if (arg == "--repeat" && i + 1 < argc) opts.repeat = std::atoi(argv[++i]);
        else if (arg == "--verbose")    opts.verbose = true;
        else if (arg == "--no-verify")  opts.verify = false;
        else if (arg == "--stats-only") opts.stats_only = true;
        else if (arg == "--tile-vol" && i + 1 < argc) opts.tile_vol = std::atoi(argv[++i]);
        else if (arg == "--scan" && i + 1 < argc) opts.scan_dir = argv[++i];
        else if (arg == "--tiler" && i + 1 < argc) {
            std::string t = argv[++i];
            if (t == "greedy")       opts.tiler = TilerMode::Greedy;
            else if (t == "search")  opts.tiler = TilerMode::Search;
            else if (t == "compare") opts.tiler = TilerMode::Compare;
            else { std::cerr << "Unknown tiler: " << t << "\n"; exit(1); }
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage:\n"
                      << "  Einsum mode: test_hypertile --a <A.bs> --b <B.bs> --einsum <expr>\n"
                      << "  Perm mode:   test_hypertile --tensor <T.bs> --perm \"2,0,1\"\n"
                      << "  Shape mode:  test_hypertile --shape 37x37x37 --perm \"1,2,0\"\n"
                      << "  Scan mode:   test_hypertile --scan <tensor_dir> [--tiler compare]\n"
                      << "\nOptions:\n"
                      << "  --dtype f32|f64         Data type (default: f32)\n"
                      << "  --warmup N              Warmup iterations (default: 3)\n"
                      << "  --repeat N              Timed iterations (default: 5)\n"
                      << "  --verbose               Per-block task details\n"
                      << "  --no-verify             Skip correctness check\n"
                      << "  --stats-only            Print stats only, no kernel launch\n"
                      << "  --tile-vol N            Hypertile tile volume (default: 4096)\n"
                      << "  --tiler greedy|search|compare  Tiler selection (default: greedy)\n"
                      << "  --scan <dir>            Scan all einsum cases in directory\n";
            exit(0);
        }
    }
    if (!opts.is_shape_mode() && !opts.is_perm_mode() && !opts.is_einsum_mode() && !opts.is_scan_mode()) {
        std::cerr << "Error: use --shape+--perm, --tensor+--perm, --a+--b+--einsum, or --scan <dir>. Use --help.\n";
        exit(1);
    }
    return opts;
}

// Naive CPU transpose for verification.
// perm[src_dim] = dest_position.
template <typename T>
void naive_transpose(const T* src, T* dst, const int32_t* dims, const int32_t* perm, int rank) {
    int64_t total = 1;
    for (int d = 0; d < rank; ++d) total *= dims[d];

    // Compute src and dst strides
    std::vector<int64_t> src_stride(rank), dst_stride(rank);
    src_stride[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; --d)
        src_stride[d] = src_stride[d + 1] * dims[d + 1];

    // Inverse perm: inv_perm[dest_pos] = src_dim
    std::vector<int32_t> inv_perm(rank);
    for (int d = 0; d < rank; ++d) inv_perm[perm[d]] = d;

    // Dest dims in dest order: dst_shape[dp] = dims[inv_perm[dp]]
    std::vector<int32_t> dst_shape(rank);
    for (int dp = 0; dp < rank; ++dp) dst_shape[dp] = dims[inv_perm[dp]];

    dst_stride[rank - 1] = 1;
    for (int dd = rank - 2; dd >= 0; --dd)
        dst_stride[dd] = dst_stride[dd + 1] * dst_shape[dd + 1];

    // Map dst_stride to src dim order
    std::vector<int64_t> dst_stride_for_src(rank);
    for (int sd = 0; sd < rank; ++sd)
        dst_stride_for_src[sd] = dst_stride[perm[sd]];

    for (int64_t i = 0; i < total; ++i) {
        int64_t rem = i;
        int64_t dst_idx = 0;
        for (int d = 0; d < rank; ++d) {
            int64_t coord = rem / src_stride[d];
            rem %= src_stride[d];
            dst_idx += coord * dst_stride_for_src[d];
        }
        dst[dst_idx] = src[i];
    }
}

// Compute tiling details for a single shape using a given tiler.
struct TilingDetail {
    int32_t tile_shape[HT_MAX_RANK];
    int volume;
    int read_tail;
    int write_tail;
    int min_tail;
    double edge_waste;
};

inline TilingDetail compute_tiling_detail(
    int32_t squashed_rank,
    const int32_t* full_shape,
    const int64_t* src_stride,
    const int64_t* dst_stride,
    int32_t tile_vol_limit,
    bool use_search)
{
    TilingDetail d{};
    if (use_search)
        optimize_tile_shape_search(squashed_rank, full_shape, src_stride, dst_stride, tile_vol_limit, d.tile_shape);
    else
        optimize_tile_shape(squashed_rank, full_shape, src_stride, dst_stride, tile_vol_limit, d.tile_shape);

    d.volume = 1;
    double edge_eff = 1.0;
    for (int i = 0; i < squashed_rank; ++i) {
        d.volume *= d.tile_shape[i];
        int ntiles = (full_shape[i] + d.tile_shape[i] - 1) / d.tile_shape[i];
        edge_eff *= static_cast<double>(full_shape[i]) / (ntiles * d.tile_shape[i]);
    }
    d.edge_waste = 1.0 - edge_eff;

    // Compute coalesced tails
    int32_t ro[HT_MAX_RANK], wo[HT_MAX_RANK];
    std::iota(ro, ro + squashed_rank, 0);
    std::sort(ro, ro + squashed_rank, [&](int a, int b) { return src_stride[a] < src_stride[b]; });
    std::iota(wo, wo + squashed_rank, 0);
    std::sort(wo, wo + squashed_rank, [&](int a, int b) { return dst_stride[a] < dst_stride[b]; });

    auto tail = [&](const int32_t* order) {
        int t = 1;
        for (int k = 0; k < squashed_rank; ++k) {
            int dim = order[k];
            t *= d.tile_shape[dim];
            if (d.tile_shape[dim] < full_shape[dim]) break;
        }
        return t;
    };
    d.read_tail = tail(ro);
    d.write_tail = tail(wo);
    d.min_tail = std::min(d.read_tail, d.write_tail);
    return d;
}

// Print one operand's transpose tiling statistics.
template <typename T>
void print_operand_stats(
    const char* name,
    const GPUSparseInput<T>& gpu,
    const GemmLayoutPlan::OperandPlan& op_plan,
    int tile_vol_limit,
    bool verbose)
{
    if (!op_plan.transpose) {
        std::cout << "  " << name << ": no transpose needed\n";
        return;
    }

    const dim_t rank = gpu.rank;
    const dim_t head_rank = rank - op_plan.unmoved_tail_dims;

    std::cout << "  " << name << ": transpose required\n"
              << "    perm: [";
    for (int d = 0; d < head_rank; ++d) {
        if (d) std::cout << ",";
        std::cout << op_plan.perm[d];
    }
    std::cout << "]  rank=" << rank << "  head_rank=" << head_rank
              << "  unmoved_tail=" << op_plan.unmoved_tail_dims << "\n";

    // Build hypertile index
    auto shape_map = build_block_shape_map(gpu.block_dims.data(), gpu.num_blocks, gpu.rank);
    auto index = build_hypertile_index(gpu, op_plan, shape_map, tile_vol_limit);

    if (!index.valid) {
        std::cout << "    (all blocks trivial — no hypertile work)\n";
        return;
    }

    int64_t ht_elements = 0;
    int trivial_blocks = 0, nontrivial_blocks = 0;
    int64_t total_tiles = 0;
    for (int b = 0; b < static_cast<int>(gpu.num_blocks); ++b) {
        ht_elements += gpu.block_volumes[b];
        uint32_t sid = shape_map.block_to_shape[b];
        const auto& entry = index.per_shape[sid];
        if (entry.trivial) trivial_blocks++;
        else { nontrivial_blocks++; total_tiles += entry.num_tiles; }
    }

    std::cout << "    blocks: " << gpu.num_blocks
              << " (nontrivial: " << nontrivial_blocks
              << ", trivial: " << trivial_blocks << ")\n"
              << "    elements: " << ht_elements << "\n"
              << "    unique_shapes: " << shape_map.num_unique
              << "  total_tiles: " << total_tiles << "\n";

    // Index-level stats
    std::cout << "    --- Index stats ---\n"
              << "    max_squashed_rank: " << index.max_squashed_rank
              << "  max_tile_volume: " << index.max_tile_volume << "\n"
              << "    min_rr: " << index.min_rr
              << "  min_wr: " << index.min_wr << "\n"
              << "    num_slices: " << index.num_slices << "\n";

    // Per-slice info
    for (int s = 0; s < index.num_slices; ++s) {
        const auto& si = index.slices[s];
        std::cout << "      slice " << s
                  << ": rank(WR)=" << si.rank
                  << " num_shapes=" << si.num_shapes
                  << " shape_bytes=" << si.shape_bytes << "\n";
    }

    // Per-shape distribution (WR)
    std::map<int, int> wr_dist;
    for (int s = 0; s < shape_map.num_unique; ++s) {
        const auto& entry = index.per_shape[s];
        if (!entry.trivial) wr_dist[entry.rank]++;
    }
    std::cout << "    --- WR distribution (unique shapes) ---\n";
    for (auto& [wr, cnt] : wr_dist)
        std::cout << "      WR=" << wr << ": " << cnt << " shapes\n";

    // Per-entry details (verbose)
    if (verbose) {
        // Build full perm + inverse for tiling detail computation
        std::vector<int32_t> full_perm(rank);
        for (int32_t d = 0; d < head_rank; ++d) full_perm[d] = op_plan.perm[d];
        for (int32_t d = head_rank; d < rank; ++d) full_perm[d] = d;
        std::vector<int32_t> inv_perm(rank);
        for (int32_t sd = 0; sd < rank; ++sd) inv_perm[full_perm[sd]] = sd;

        std::cout << "    --- Per-shape details ---\n";
        for (int s = 0; s < shape_map.num_unique; ++s) {
            const auto& entry = index.per_shape[s];
            const dim_t* dims = shape_map.shape_dims(static_cast<uint32_t>(s));

            // Count blocks with this shape
            int block_count = 0;
            for (int b = 0; b < static_cast<int>(gpu.num_blocks); ++b)
                if (shape_map.block_to_shape[b] == static_cast<uint32_t>(s)) block_count++;

            std::cout << "      shape " << s << ": dims=[";
            for (int d = 0; d < shape_map.rank; ++d) {
                if (d) std::cout << "x";
                std::cout << dims[d];
            }
            std::cout << "] blocks=" << block_count;
            if (entry.trivial) {
                std::cout << " (trivial)\n";
            } else {
                std::cout << " WR=" << static_cast<int>(entry.rank)
                          << " slice=" << static_cast<int>(entry.slice_idx)
                          << " shape_idx=" << entry.shape_idx
                          << " tiles=" << entry.num_tiles;

                // Compute and print tile shape
                auto task_result = build_task_for_shape(
                    dims, rank, head_rank,
                    full_perm.data(), inv_perm.data(), tile_vol_limit);
                if (!task_result.skip) {
                    auto& info = task_result.info;
                    int sq = info.squashed_rank;
                    auto detail = compute_tiling_detail(
                        sq, info.full_shape, info.src_stride, info.dst_stride,
                        tile_vol_limit, g_use_search_tiler);
                    std::cout << " sq=[";
                    for (int d = 0; d < sq; ++d) {
                        if (d) std::cout << "x";
                        std::cout << info.full_shape[d];
                    }
                    std::cout << "] tile=[";
                    for (int d = 0; d < sq; ++d) {
                        if (d) std::cout << "x";
                        std::cout << detail.tile_shape[d];
                    }
                    std::cout << "] rt=" << detail.read_tail
                              << " wt=" << detail.write_tail
                              << " ew=" << std::fixed << std::setprecision(2)
                              << detail.edge_waste << std::defaultfloat;
                }
                std::cout << "\n";
            }
        }
    }
}

// Compute the free-dimension product for a block of a given operand.
// For operand A this is M; for operand B this is N.
inline dim_t block_free_dim_product(
    const dim_t* block_dims, dim_t rank, int64_t blk_idx,
    const int32_t* free_pos, dim_t num_free)
{
    dim_t prod = 1;
    for (int32_t i = 0; i < num_free; ++i)
        prod *= block_dims[blk_idx * rank + free_pos[i]];
    return prod;
}

// Process one operand's transpose: build precomp, run kernel, verify, report timing.
// When config is provided with the other operand's max free-dim product, blocks that
// can never appear in a CUTLASS-tier pair are excluded (warp/volume handle layout natively).
template <typename T>
int process_operand(
    const char* name,
    const GPUSparseInput<T>& gpu,
    const GemmLayoutPlan::OperandPlan& op_plan,
    const Options& opts,
    const ContractionConfig* config = nullptr,
    bool is_operand_A = true,
    dim_t other_max_free = 0)
{
    if (!op_plan.transpose || op_plan.dispatch_kind != TransposeDispatchKind::GenericPermute)
        return 0;

    const dim_t rank = gpu.rank;
    const dim_t head_rank = rank - op_plan.unmoved_tail_dims;
    const int n = static_cast<int>(gpu.num_blocks);

    // Upload source data
    const T* h_src = gpu.values.empty() ? gpu.values_ptr : gpu.values.data();
    int64_t total_elems = gpu.values.empty() ? gpu.total_values : static_cast<int64_t>(gpu.values.size());

    T* d_src = nullptr;
    T* d_dst = nullptr;
    check_cuda(cudaMalloc(&d_src, total_elems * sizeof(T)), "malloc d_src");
    check_cuda(cudaMalloc(&d_dst, total_elems * sizeof(T)), "malloc d_dst");
    check_cuda(cudaMemcpy(d_src, h_src, total_elems * sizeof(T), cudaMemcpyHostToDevice), "cpy src");
    check_cuda(cudaMemset(d_dst, 0, total_elems * sizeof(T)), "memset dst");

    // Build precomp metadata
    cudaStream_t stream, tiled_stream;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&tiled_stream);

    auto smap = build_block_shape_map(gpu.block_dims.data(), gpu.num_blocks, gpu.rank);
    auto idx = build_hypertile_index<T>(gpu, op_plan, smap);
    auto triv = compute_trivial_shapes(smap, &idx);

    // Filter blocks: only include those that could appear in a CUTLASS-tier pair.
    // Tier classification per pair: warp (min(M,N)<4), volume (tile_count<=256 + K check),
    // CUTLASS (everything else).  Only CUTLASS needs pre-transposed data.
    //
    // For operand A with free_prod=M:
    //   - M < 4: all pairs have min(M,N)<4 → warp tier → skip
    //   - M >= 4 AND other_max_free > 32: exists a pair with N>32 → CUTLASS → include
    //   - M >= 4 AND M <= 32 AND other_max_free <= 32: check volume threshold with K
    std::vector<int> blocks;
    int skipped_small = 0;
    if (config && other_max_free > 0) {
        const int32_t* free_pos = is_operand_A ? config->free_pos_A : config->free_pos_B;
        dim_t num_free = is_operand_A ? config->num_free_A : config->num_free_B;
        const int32_t* contr_pos = is_operand_A ? config->contracted_pos_A : config->contracted_pos_B;
        dim_t num_contr = config->num_contracted;
        for (int b = 0; b < n; ++b) {
            dim_t free_prod = block_free_dim_product(
                gpu.block_dims.data(), rank, b, free_pos, num_free);
            // Warp tier: min(M,N) < 4 AND max(M,N) <= 1024
            // free_prod is this operand's dim; other_max_free is the worst-case other dim.
            // Only skip if max(free_prod, other_max_free) <= MAX_WARP_DIM.
            if (free_prod < MIN_CUTLASS_DIM &&
                std::max(free_prod, other_max_free) <= MAX_WARP_DIM) {
                skipped_small++;
                continue;
            }
            // Volume tier: ceil(M/2)*ceil(N/2) <= 256 AND M*K+N*K <= vol_threshold
            if (vol_tile_count(free_prod, other_max_free) <= VOL_MAX_TILES) {
                dim_t K = block_free_dim_product(
                    gpu.block_dims.data(), rank, b, contr_pos, num_contr);
                int64_t mk_nk = static_cast<int64_t>(free_prod) * K
                               + static_cast<int64_t>(other_max_free) * K;
                if (mk_nk <= vol_threshold<T>()) {
                    skipped_small++;
                    continue;
                }
            }
            blocks.push_back(b);
        }
    } else {
        blocks.resize(n);
        std::iota(blocks.begin(), blocks.end(), 0);
    }

    TransposeFlatMeta meta = assemble_hypertile_batch<T>(
        blocks, gpu, smap, idx, triv, {}, nullptr, stream);
    cudaStreamSynchronize(stream);

    std::cout << "\n=== " << name << " Transpose Kernel ===\n"
              << "  hypertile: tasks=" << [&]{ int t=0; for(int i=0;i<meta.ht_num_slices;++i) t+=meta.ht_slices[i].num_tasks; return t; }()
              << " tiles=" << meta.ht_total_tiles;
    if (skipped_small > 0)
        std::cout << "  (skipped " << skipped_small << " small blocks)";
    std::cout << "\n";

    // Warmup
    for (int w = 0; w < opts.warmup; ++w) {
        check_cuda(cudaMemset(d_dst, 0, total_elems * sizeof(T)), "memset");
        launch_batched_transpose<T>(d_src, d_dst, meta, tiled_stream);
        cudaDeviceSynchronize();
    }

    // Timed runs
    cudaEvent_t ev_start, ev_stop, ev_tiled_done;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventCreate(&ev_tiled_done);

    std::vector<float> times;
    for (int r = 0; r < opts.repeat; ++r) {
        check_cuda(cudaMemset(d_dst, 0, total_elems * sizeof(T)), "memset");
        cudaEventRecord(ev_start, stream);
        cudaStreamWaitEvent(tiled_stream, ev_start, 0);
        launch_batched_transpose<T>(d_src, d_dst, meta, tiled_stream);
        cudaEventRecord(ev_tiled_done, tiled_stream);
        cudaStreamWaitEvent(stream, ev_tiled_done, 0);
        cudaEventRecord(ev_stop, stream);
        cudaStreamSynchronize(stream);
        cudaStreamSynchronize(tiled_stream);
        float ms = 0;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());
    float median_ms = times[times.size() / 2];

    // Bandwidth: only count blocks actually transposed (nontrivial + in filtered set)
    int64_t nontrivial_elems = 0;
    for (int bi : blocks) {
        uint32_t sid = smap.block_to_shape[bi];
        if (!triv[sid])
            nontrivial_elems += gpu.block_volumes[bi];
    }
    int64_t transpose_bytes = nontrivial_elems * sizeof(T) * 2;  // read + write
    double gb_s = (static_cast<double>(transpose_bytes) / 1e9) / (median_ms / 1e3);

    std::cout << "  timing: median=" << std::fixed << std::setprecision(2) << median_ms
              << "ms  min=" << times.front() << "ms  max=" << times.back() << "ms"
              << "  BW=" << std::setprecision(1) << gb_s << " GB/s\n"
              << std::defaultfloat;

    // Verification — only check filtered nontrivial blocks.
    int errors = 0;
    if (opts.verify) {
        // Full-rank perm
        std::vector<int32_t> full_perm(rank);
        for (int32_t d = 0; d < head_rank; ++d) full_perm[d] = op_plan.perm[d];
        for (int32_t d = head_rank; d < rank; ++d) full_perm[d] = d;

        // Copy GPU result back
        std::vector<T> h_dst(total_elems);
        check_cuda(cudaMemcpy(h_dst.data(), d_dst, total_elems * sizeof(T),
                              cudaMemcpyDeviceToHost), "cpy result");

        // Run naive reference per block (filtered nontrivial only)
        std::vector<T> h_ref(total_elems, T(0));
        for (int bi : blocks) {
            uint32_t sid = smap.block_to_shape[bi];
            if (triv[sid]) continue;
            const dim_t* dims = &gpu.block_dims[bi * rank];
            int64_t offset = gpu.block_offsets[bi];

            std::vector<int32_t> bdims(rank);
            for (int d = 0; d < rank; ++d) bdims[d] = dims[d];

            naive_transpose<T>(h_src + offset, h_ref.data() + offset,
                               bdims.data(), full_perm.data(), rank);
        }

        // Compare filtered nontrivial blocks only
        int block_errors = 0;
        int verified = 0;
        for (int bi : blocks) {
            uint32_t sid = smap.block_to_shape[bi];
            if (triv[sid]) continue;
            verified++;
            int64_t offset = gpu.block_offsets[bi];
            int64_t vol = gpu.block_volumes[bi];
            bool block_ok = true;
            for (int64_t i = 0; i < vol; ++i) {
                T got = h_dst[offset + i];
                T expected = h_ref[offset + i];
                if (got != expected) {
                    if (errors < 10) {
                        std::cerr << "  MISMATCH block " << bi
                                  << " elem " << i << ": got=" << got
                                  << " expected=" << expected << "\n";
                    }
                    errors++;
                    block_ok = false;
                }
            }
            if (!block_ok) block_errors++;
        }
        int skipped = n - verified;
        if (errors == 0) {
            std::cout << "  verification: PASS (" << verified << " blocks verified, "
                      << skipped << " trivial/small skipped)\n";
        } else {
            std::cout << "  verification: FAIL (" << errors << " mismatches in "
                      << block_errors << "/" << verified << " blocks)\n";
        }
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaEventDestroy(ev_tiled_done);
    cudaStreamDestroy(stream);
    cudaStreamDestroy(tiled_stream);
    meta.free_gpu();
    cudaFree(d_src);
    cudaFree(d_dst);

    return errors;
}

// --- Shape mode: synthetic single block with given dims + perm ---
template <typename T>
int run_shape(const Options& opts) {
    auto dims = parse_shape_string(opts.shape_str);
    auto perm = parse_perm_string(opts.perm_str);
    const dim_t rank = static_cast<dim_t>(dims.size());

    if (static_cast<int>(perm.size()) != rank) {
        std::cerr << "Error: perm length (" << perm.size()
                  << ") != shape rank (" << rank << ")\n";
        return 1;
    }
    std::vector<bool> seen(rank, false);
    for (int32_t d : perm) {
        if (d < 0 || d >= rank || seen[d]) {
            std::cerr << "Error: invalid permutation\n";
            return 1;
        }
        seen[d] = true;
    }

    int64_t volume = 1;
    for (int d = 0; d < rank; ++d) volume *= dims[d];

    std::cout << "Synthetic shape: ";
    for (int d = 0; d < rank; ++d) { if (d) std::cout << "x"; std::cout << dims[d]; }
    std::cout << "  perm: [";
    for (int d = 0; d < rank; ++d) { if (d) std::cout << ","; std::cout << perm[d]; }
    std::cout << "]  volume: " << volume
              << " (" << std::fixed << std::setprecision(1)
              << (volume * sizeof(T) / 1e6) << " MB)\n" << std::defaultfloat;

    // Build synthetic GPUSparseInput with one block
    GPUSparseInput<T> gpu;
    gpu.rank = rank;
    gpu.num_blocks = 1;
    gpu.block_dims.assign(dims.begin(), dims.end());
    gpu.block_coords.assign(rank, 0);
    gpu.block_offsets = {0};
    gpu.block_volumes = {volume};
    gpu.total_values = volume;
    gpu.labels.resize(rank);
    std::iota(gpu.labels.begin(), gpu.labels.end(), 'a');
    gpu.full_extents.assign(dims.begin(), dims.end());
    // block_sizes: one block per dim, size = full extent
    gpu.num_blocks_per_dim.assign(rank, 1);
    gpu.block_sizes_offsets.resize(rank);
    gpu.block_sizes_flat.resize(rank);
    for (int d = 0; d < rank; ++d) {
        gpu.block_sizes_offsets[d] = d;
        gpu.block_sizes_flat[d] = dims[d];
    }

    // Build OperandPlan
    GemmLayoutPlan::OperandPlan op_plan;
    op_plan.transpose = true;
    op_plan.dispatch_kind = TransposeDispatchKind::GenericPermute;
    op_plan.unmoved_tail_dims = 0;
    op_plan.perm = perm;
    while (op_plan.unmoved_tail_dims < rank &&
           perm[rank - 1 - op_plan.unmoved_tail_dims] == rank - 1 - op_plan.unmoved_tail_dims)
        op_plan.unmoved_tail_dims++;

    dim_t head_rank = rank - op_plan.unmoved_tail_dims;
    if (head_rank < 2) {
        std::cout << "Perm is identity (or nearly so) — no transpose needed.\n";
        return 0;
    }
    op_plan.perm.resize(head_rank);
    std::cout << "  head_rank=" << head_rank
              << "  unmoved_tail=" << op_plan.unmoved_tail_dims << "\n";

    // Stats
    std::cout << "\n=== Hypertile Tiling Statistics (tile_vol=" << opts.tile_vol << ") ===\n";
    print_operand_stats<T>("T", gpu, op_plan, opts.tile_vol, opts.verbose);

    if (opts.tile_vol != HT_TILE_VOL) {
        std::cout << "\n=== Default Tiling Statistics (tile_vol=" << HT_TILE_VOL << ") ===\n";
        print_operand_stats<T>("T", gpu, op_plan, HT_TILE_VOL, opts.verbose);
    }

    if (opts.stats_only) return 0;

    // Allocate synthetic data for kernel run + verification
    gpu.values.resize(volume);
    for (int64_t i = 0; i < volume; ++i) gpu.values[i] = static_cast<T>(i % 1000);

    return process_operand<T>("T", gpu, op_plan, opts);
}

// --- Perm mode: single tensor + explicit permutation ---
template <typename T>
int run_perm(const Options& opts) {
    auto perm = parse_perm_string(opts.perm_str);

    std::cout << "Loading tensor...\n";
    auto tensors = parse_tensor_files({opts.tensor_path}, opts.dtype);
    BlockSparseTensor& tensor = tensors[0];
    auto gpu = to_gpu_input<T>(tensor);

    const dim_t rank = gpu.rank;
    if (static_cast<int>(perm.size()) != rank) {
        std::cerr << "Error: perm length (" << perm.size()
                  << ") != tensor rank (" << rank << ")\n";
        return 1;
    }

    // Validate perm is a valid permutation
    std::vector<bool> seen(rank, false);
    for (int32_t d : perm) {
        if (d < 0 || d >= rank || seen[d]) {
            std::cerr << "Error: invalid permutation\n";
            return 1;
        }
        seen[d] = true;
    }

    std::cout << "  tensor: "
              << std::string(tensor.meta.indices.begin(), tensor.meta.indices.end())
              << " rank=" << rank << " blocks=" << gpu.num_blocks
              << " elements=" << gpu.total_values << "\n"
              << "  perm: [";
    for (int d = 0; d < rank; ++d) {
        if (d) std::cout << ",";
        std::cout << perm[d];
    }
    std::cout << "]\n";

    // Build a synthetic OperandPlan from the raw perm
    GemmLayoutPlan::OperandPlan op_plan;
    op_plan.transpose = true;
    op_plan.dispatch_kind = TransposeDispatchKind::GenericPermute;
    op_plan.unmoved_tail_dims = 0;
    op_plan.perm = perm;

    // Count unmoved tail dims (trailing identity suffix)
    while (op_plan.unmoved_tail_dims < rank &&
           perm[rank - 1 - op_plan.unmoved_tail_dims] == rank - 1 - op_plan.unmoved_tail_dims)
        op_plan.unmoved_tail_dims++;

    dim_t head_rank = rank - op_plan.unmoved_tail_dims;
    if (head_rank < 2) {
        std::cout << "Perm is identity (or nearly so) — no transpose needed.\n";
        return 0;
    }
    // Trim perm to head dims only
    op_plan.perm.resize(head_rank);

    std::cout << "  head_rank=" << head_rank
              << "  unmoved_tail=" << op_plan.unmoved_tail_dims << "\n";

    // Stats
    std::cout << "\n=== Hypertile Tiling Statistics (tile_vol=" << HT_TILE_VOL << ") ===\n";
    print_operand_stats<T>("T", gpu, op_plan, HT_TILE_VOL, opts.verbose);

    if (opts.tile_vol != HT_TILE_VOL) {
        std::cout << "\n=== What-if Tiling Statistics (tile_vol=" << opts.tile_vol << ") ===\n";
        print_operand_stats<T>("T", gpu, op_plan, opts.tile_vol, opts.verbose);
    }

    if (opts.stats_only) return 0;

    return process_operand<T>("T", gpu, op_plan, opts);
}

// --- Einsum mode: two tensors + einsum expression ---
template <typename T>
int run_einsum(const Options& opts) {
    std::cout << "Loading tensors...\n";
    auto tensors = parse_tensor_files({opts.tensor_a, opts.tensor_b}, opts.dtype);
    BlockSparseTensor& A = tensors[0];
    BlockSparseTensor& B = tensors[1];

    auto A_gpu = to_gpu_input<T>(A);
    auto B_gpu = to_gpu_input<T>(B);

    std::cout << "  A: " << std::string(A.meta.indices.begin(), A.meta.indices.end())
              << " rank=" << A_gpu.rank << " blocks=" << A_gpu.num_blocks
              << " elements=" << A_gpu.total_values << "\n";
    std::cout << "  B: " << std::string(B.meta.indices.begin(), B.meta.indices.end())
              << " rank=" << B_gpu.rank << " blocks=" << B_gpu.num_blocks
              << " elements=" << B_gpu.total_values << "\n";

    auto expr = parse_einsum(opts.einsum_str);
    std::vector<std::vector<int64_t>> shapes = {
        block_utils::compute_full_shape(A),
        block_utils::compute_full_shape(B)
    };
    validate_einsum(expr, shapes);

    EinsumIndexSets index_sets;
    create_einsum_index_sets(expr, index_sets);

    std::cout << "Einsum: " << opts.einsum_str << "\n"
              << "  free_A: ";
    for (char c : index_sets.free_indices_A) std::cout << c;
    std::cout << "  free_B: ";
    for (char c : index_sets.free_indices_B) std::cout << c;
    std::cout << "  contracted: ";
    for (char c : index_sets.contracted_indices) std::cout << c;
    std::cout << "  batch: ";
    for (char c : index_sets.batch_indices) std::cout << c;
    std::cout << "\n";

    ContractionConfig config = build_contraction_config(
        index_sets,
        std::vector<char>(A_gpu.labels.begin(), A_gpu.labels.end()),
        std::vector<char>(B_gpu.labels.begin(), B_gpu.labels.end()),
        expr.output_labels);
    GemmLayoutPlan layout_plan = plan_gemm_layout(config, A_gpu.rank, B_gpu.rank);

    std::cout << "\nLayout plan:\n"
              << "  GEMM mode after transpose: " << gemm_mode_to_string(layout_plan.mode_after_transpose) << "\n"
              << "  A: " << (layout_plan.layout_A == GemmOperandLayout::Normal ? "Normal" :
                            layout_plan.layout_A == GemmOperandLayout::Transposed ? "Transposed" : "Interleaved")
              << " -> " << (layout_plan.plan_A.target_layout == GemmOperandLayout::Normal ? "Normal" : "Transposed")
              << (layout_plan.plan_A.transpose ? " (TRANSPOSE)" : "") << "\n"
              << "  B: " << (layout_plan.layout_B == GemmOperandLayout::Normal ? "Normal" :
                            layout_plan.layout_B == GemmOperandLayout::Transposed ? "Transposed" : "Interleaved")
              << " -> " << (layout_plan.plan_B.target_layout == GemmOperandLayout::Normal ? "Normal" : "Transposed")
              << (layout_plan.plan_B.transpose ? " (TRANSPOSE)" : "") << "\n";

    std::cout << "\n=== Hypertile Tiling Statistics (tile_vol=" << HT_TILE_VOL << ") ===\n";
    print_operand_stats<T>("A", A_gpu, layout_plan.plan_A, HT_TILE_VOL, opts.verbose);
    print_operand_stats<T>("B", B_gpu, layout_plan.plan_B, HT_TILE_VOL, opts.verbose);

    if (opts.tile_vol != HT_TILE_VOL) {
        std::cout << "\n=== What-if Tiling Statistics (tile_vol=" << opts.tile_vol << ") ===\n";
        print_operand_stats<T>("A", A_gpu, layout_plan.plan_A, opts.tile_vol, opts.verbose);
        print_operand_stats<T>("B", B_gpu, layout_plan.plan_B, opts.tile_vol, opts.verbose);
    }

    if (opts.stats_only) return 0;

    // Compute max free-dim product for each operand (used to filter the other).
    // max_M = max over A blocks of product(free_A dims)
    // max_N = max over B blocks of product(free_B dims)
    dim_t max_M = 0, max_N = 0;
    for (int b = 0; b < static_cast<int>(A_gpu.num_blocks); ++b)
        max_M = std::max(max_M, block_free_dim_product(
            A_gpu.block_dims.data(), A_gpu.rank, b, config.free_pos_A, config.num_free_A));
    for (int b = 0; b < static_cast<int>(B_gpu.num_blocks); ++b)
        max_N = std::max(max_N, block_free_dim_product(
            B_gpu.block_dims.data(), B_gpu.rank, b, config.free_pos_B, config.num_free_B));

    int total_errors = 0;
    total_errors += process_operand<T>("A", A_gpu, layout_plan.plan_A, opts, &config, true, max_N);
    total_errors += process_operand<T>("B", B_gpu, layout_plan.plan_B, opts, &config, false, max_M);

    if (!layout_plan.plan_A.transpose && !layout_plan.plan_B.transpose) {
        std::cout << "\nNo transpose needed for this contraction.\n";
    }

    return total_errors > 0 ? 1 : 0;
}

// Parse info.txt for einsum string
std::string parse_info_einsum(const std::string& path) {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("einsum:", 0) == 0) {
            auto pos = line.find_first_not_of(" \t", 7);
            return (pos != std::string::npos) ? line.substr(pos) : "";
        }
    }
    return "";
}

struct BenchResult {
    double gb_s;
    double host_us;  // host-side tiler + index build time in microseconds
};

// Benchmark one operand's transpose with a specific tiler.
// Returns bandwidth in GB/s and host overhead, or {0,0} if no transpose needed.
template <typename T>
BenchResult bench_operand_transpose(
    const GPUSparseInput<T>& gpu,
    const GemmLayoutPlan::OperandPlan& op_plan,
    bool use_search,
    int warmup, int repeat)
{
    double host_us = 0.0;
    if (!op_plan.transpose || op_plan.dispatch_kind != TransposeDispatchKind::GenericPermute)
        return {0.0, 0.0};

    const T* h_src = gpu.values.empty() ? gpu.values_ptr : gpu.values.data();
    int64_t total_elems = gpu.values.empty() ? gpu.total_values : static_cast<int64_t>(gpu.values.size());
    if (total_elems == 0) return {0.0, 0.0};

    T* d_src = nullptr;
    T* d_dst = nullptr;
    cudaMalloc(&d_src, total_elems * sizeof(T));
    cudaMalloc(&d_dst, total_elems * sizeof(T));
    cudaMemcpy(d_src, h_src, total_elems * sizeof(T), cudaMemcpyHostToDevice);

    cudaStream_t stream, tiled_stream;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&tiled_stream);

    g_use_search_tiler = use_search;
    auto t_host_start = std::chrono::high_resolution_clock::now();
    auto smap = build_block_shape_map(gpu.block_dims.data(), gpu.num_blocks, gpu.rank);
    auto idx = build_hypertile_index<T>(gpu, op_plan, smap);
    auto triv = compute_trivial_shapes(smap, &idx);
    auto t_host_end = std::chrono::high_resolution_clock::now();
    host_us = std::chrono::duration<double, std::micro>(t_host_end - t_host_start).count();

    std::vector<int> all_blocks(gpu.num_blocks);
    std::iota(all_blocks.begin(), all_blocks.end(), 0);
    TransposeFlatMeta meta = assemble_hypertile_batch<T>(
        all_blocks, gpu, smap, idx, triv, {}, nullptr, stream);
    cudaStreamSynchronize(stream);
    g_use_search_tiler = false;

    // Warmup
    for (int w = 0; w < warmup; ++w) {
        cudaMemset(d_dst, 0, total_elems * sizeof(T));
        launch_batched_transpose<T>(d_src, d_dst, meta, tiled_stream);
        cudaDeviceSynchronize();
    }

    // L2 flush buffer: write 48 MB of garbage to evict cached data
    constexpr size_t FLUSH_SIZE = 48 * 1024 * 1024;
    void* d_flush = nullptr;
    cudaMalloc(&d_flush, FLUSH_SIZE);

    // Timed runs
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    std::vector<float> times;
    for (int r = 0; r < repeat; ++r) {
        // Flush L2 cache before each iteration
        cudaMemset(d_flush, r + 1, FLUSH_SIZE);
        cudaMemset(d_dst, 0, total_elems * sizeof(T));
        cudaStreamSynchronize(tiled_stream);

        cudaEventRecord(ev_start, tiled_stream);
        launch_batched_transpose<T>(d_src, d_dst, meta, tiled_stream);
        cudaEventRecord(ev_stop, tiled_stream);
        cudaStreamSynchronize(tiled_stream);
        float ms = 0;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        times.push_back(ms);
    }
    cudaFree(d_flush);

    std::sort(times.begin(), times.end());
    float median_ms = times[times.size() / 2];
    double gb_s = (static_cast<double>(total_elems) * sizeof(T) * 2.0 / 1e9) / (median_ms / 1e3);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaStreamDestroy(stream);
    cudaStreamDestroy(tiled_stream);
    meta.free_gpu();
    cudaFree(d_src);
    cudaFree(d_dst);

    return {gb_s, host_us};
}

// Compare tiling for one operand across both tilers.
// Returns per-shape comparison lines for summary table.
template <typename T>
void compare_operand_tiling(
    const char* op_name,
    const std::string& case_name,
    const GPUSparseInput<T>& gpu,
    const GemmLayoutPlan::OperandPlan& op_plan,
    int tile_vol_limit)
{
    if (!op_plan.transpose) return;

    const dim_t rank = gpu.rank;
    const dim_t head_rank = rank - op_plan.unmoved_tail_dims;

    // Build full-rank perm + inverse
    std::vector<int32_t> full_perm(rank);
    for (int32_t d = 0; d < head_rank; ++d) full_perm[d] = op_plan.perm[d];
    for (int32_t d = head_rank; d < rank; ++d) full_perm[d] = d;
    std::vector<int32_t> inv_perm(rank);
    for (int32_t sd = 0; sd < rank; ++sd)
        inv_perm[full_perm[sd]] = sd;

    auto shape_map = build_block_shape_map(gpu.block_dims.data(), gpu.num_blocks, gpu.rank);

    for (int s = 0; s < shape_map.num_unique; ++s) {
        const dim_t* dims = shape_map.shape_dims(static_cast<uint32_t>(s));

        // Count blocks with this shape
        int block_count = 0;
        for (int b = 0; b < static_cast<int>(gpu.num_blocks); ++b)
            if (shape_map.block_to_shape[b] == static_cast<uint32_t>(s)) block_count++;

        // Run build_task_for_shape to get squashed shape + strides
        auto task_result = build_task_for_shape(
            dims, rank, head_rank,
            full_perm.data(), inv_perm.data(), tile_vol_limit);
        if (task_result.skip) continue;

        auto& info = task_result.info;
        int sq = info.squashed_rank;
        int32_t* sq_shape = info.full_shape;
        int64_t* sq_src = info.src_stride;
        int64_t* sq_dst = info.dst_stride;

        auto dg = compute_tiling_detail(sq, sq_shape, sq_src, sq_dst, tile_vol_limit, false);
        auto ds = compute_tiling_detail(sq, sq_shape, sq_src, sq_dst, tile_vol_limit, true);

        // Print shape dims
        std::cout << std::left << std::setw(38) << case_name
                  << " " << op_name << "  s" << s
                  << " blks=" << std::setw(4) << block_count
                  << " sq=[";
        for (int d = 0; d < sq; ++d) {
            if (d) std::cout << "x";
            std::cout << sq_shape[d];
        }
        std::cout << "]";

        // Greedy tile
        std::cout << "  G=[";
        for (int d = 0; d < sq; ++d) {
            if (d) std::cout << "x";
            std::cout << dg.tile_shape[d];
        }
        std::cout << "]"
                  << " rt=" << std::setw(4) << dg.read_tail
                  << " wt=" << std::setw(4) << dg.write_tail
                  << " ew=" << std::fixed << std::setprecision(2) << dg.edge_waste;

        // Search tile
        std::cout << "  S=[";
        for (int d = 0; d < sq; ++d) {
            if (d) std::cout << "x";
            std::cout << ds.tile_shape[d];
        }
        std::cout << "]"
                  << " rt=" << std::setw(4) << ds.read_tail
                  << " wt=" << std::setw(4) << ds.write_tail
                  << " ew=" << std::fixed << std::setprecision(2) << ds.edge_waste;

        // Flag differences
        if (dg.min_tail != ds.min_tail || dg.volume != ds.volume)
            std::cout << "  <-- DIFF";
        std::cout << std::defaultfloat << "\n";
    }
}

// Scan all einsum cases in a directory, comparing tiler outputs.
template <typename T>
int run_scan(const Options& opts) {
    std::vector<std::string> case_dirs;
    for (const auto& entry : fs::directory_iterator(opts.scan_dir)) {
        if (!entry.is_directory()) continue;
        std::string dirname = entry.path().filename().string();
        if (dirname.find("_E") == std::string::npos &&
            dirname.find("_S") == std::string::npos)
            continue;
        case_dirs.push_back(dirname);
    }
    std::sort(case_dirs.begin(), case_dirs.end());

    std::cout << "Scanning " << case_dirs.size() << " cases in " << opts.scan_dir << "\n";
    std::cout << "Tiler: " << (opts.tiler == TilerMode::Compare ? "compare (greedy vs search)" :
                                opts.tiler == TilerMode::Search ? "search" : "greedy") << "\n\n";

    for (const auto& dirname : case_dirs) {
        std::string dir = opts.scan_dir + "/" + dirname;
        std::string info_path = dir + "/" + dirname + "_info.txt";
        std::string a_path = dir + "/" + dirname + "_A.bs";
        std::string b_path = dir + "/" + dirname + "_B.bs";

        if (!fs::exists(info_path) || !fs::exists(a_path) || !fs::exists(b_path))
            continue;

        std::string einsum = parse_info_einsum(info_path);
        if (einsum.empty()) continue;

        // Load tensors
        std::vector<BlockSparseTensor> tensors;
        try {
            tensors = parse_tensor_files({a_path, b_path}, opts.dtype);
        } catch (...) {
            std::cerr << "  [skip] " << dirname << ": parse error\n";
            continue;
        }

        auto A_gpu = to_gpu_input<T>(tensors[0]);
        auto B_gpu = to_gpu_input<T>(tensors[1]);

        auto expr = parse_einsum(einsum);
        std::vector<std::vector<int64_t>> shapes = {
            block_utils::compute_full_shape(tensors[0]),
            block_utils::compute_full_shape(tensors[1])
        };
        try {
            validate_einsum(expr, shapes);
        } catch (...) {
            std::cerr << "  [skip] " << dirname << ": einsum validation error\n";
            continue;
        }

        EinsumIndexSets index_sets;
        create_einsum_index_sets(expr, index_sets);

        ContractionConfig config = build_contraction_config(
            index_sets,
            std::vector<char>(A_gpu.labels.begin(), A_gpu.labels.end()),
            std::vector<char>(B_gpu.labels.begin(), B_gpu.labels.end()),
            expr.output_labels);
        GemmLayoutPlan layout_plan = plan_gemm_layout(config, A_gpu.rank, B_gpu.rank);

        if (opts.stats_only) {
            compare_operand_tiling<T>("A", dirname, A_gpu, layout_plan.plan_A, opts.tile_vol);
            compare_operand_tiling<T>("B", dirname, B_gpu, layout_plan.plan_B, opts.tile_vol);
        } else {
            // GPU benchmark: run transpose with both tilers
            auto bench_op = [&](const char* op, const GPUSparseInput<T>& gpu,
                                GemmLayoutPlan::OperandPlan plan) {
                if (!plan.transpose) return;
                // Force GenericPermute for hypertile-only benchmarking
                plan.dispatch_kind = TransposeDispatchKind::GenericPermute;
                // Check for prior CUDA errors
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    std::cerr << "  [skip] " << dirname << " " << op
                              << ": prior CUDA error: " << cudaGetErrorString(err) << "\n";
                    return;
                }
                // Check available memory
                size_t free_mem, total_mem;
                cudaMemGetInfo(&free_mem, &total_mem);
                int64_t total_elems = gpu.values.empty()
                    ? gpu.total_values
                    : static_cast<int64_t>(gpu.values.size());
                size_t needed = total_elems * sizeof(T) * 2 + (64 << 20);  // src+dst + 64MB margin
                if (needed > free_mem) {
                    std::cout << std::left << std::setw(50) << dirname
                              << " " << op << "  SKIPPED (OOM)\n";
                    return;
                }
                auto rg = bench_operand_transpose<T>(gpu, plan, false, opts.warmup, opts.repeat);
                auto rs = bench_operand_transpose<T>(gpu, plan, true, opts.warmup, opts.repeat);
                double pct = (rg.gb_s > 0) ? (rs.gb_s - rg.gb_s) / rg.gb_s * 100.0 : 0.0;
                const char* flag = (pct > 3.0) ? " ++ SEARCH" : (pct < -3.0) ? " ** GREEDY" : "";
                std::cout << std::left << std::setw(50) << dirname
                          << " " << op
                          << "  greedy=" << std::fixed << std::setprecision(1)
                          << std::setw(7) << rg.gb_s
                          << "  search=" << std::setw(7) << rs.gb_s
                          << "  " << std::showpos << std::setprecision(1)
                          << pct << "%" << std::noshowpos << flag
                          << "  host: " << std::setprecision(0)
                          << rg.host_us << "/" << rs.host_us << "us"
                          << std::defaultfloat << "\n";
                std::cout.flush();
            };
            bench_op("A", A_gpu, layout_plan.plan_A);
            bench_op("B", B_gpu, layout_plan.plan_B);
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    Options opts = parse_args(argc, argv);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "GPU: " << prop.name << " (SM " << prop.major << "." << prop.minor
              << ", " << prop.totalGlobalMem / (1024*1024) << " MB)\n\n";

    int rc;
    if (opts.is_scan_mode()) {
        if (opts.dtype == DataType::FLOAT64)
            rc = run_scan<double>(opts);
        else
            rc = run_scan<float>(opts);
    } else if (opts.is_shape_mode()) {
        if (opts.dtype == DataType::FLOAT64)
            rc = run_shape<double>(opts);
        else
            rc = run_shape<float>(opts);
    } else if (opts.is_perm_mode()) {
        if (opts.dtype == DataType::FLOAT64)
            rc = run_perm<double>(opts);
        else
            rc = run_perm<float>(opts);
    } else {
        if (opts.dtype == DataType::FLOAT64)
            rc = run_einsum<double>(opts);
        else
            rc = run_einsum<float>(opts);
    }

    return rc;
}
