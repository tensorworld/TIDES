// fuzz_pipeline.cu — Random contraction fuzzer and scaling benchmark
//
// Modes:
//   --fuzz   Continuous fuzzing with random problems + correctness verification
//   --scale  Scaling study: sweep one parameter, report timing vs cuTENSOR

#include "general_sparse/cutensor_block_sparse.cuh"
#include "general_sparse/general_sparse_executor.cuh"
#include "general_sparse/gpu_sparse_input.hpp"
#include "general_sparse/random_contraction.hpp"
#include "general_sparse/stride_verifier.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

using namespace bsptc;

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
static bool g_scale_mode = false;
static uint64_t g_seed = 0; // 0 = time-based
static int g_count = 1000;
static int g_max_rank = 5;
static int g_max_sections = 5;
static int g_max_section_size = 64;
static double g_density = 0.3;
static bool g_f64 = false;
static bool g_stop_on_fail = false;
static int64_t g_mn_threshold = DEFAULT_CUTLASS_MN_THRESHOLD;
static bool g_verbose = false;
static int64_t g_mem_limit_bytes = 2048LL * 1024 * 1024; // 2 GB
static int64_t g_streaming_cap_mb = 0; // 0 = disabled; >0 = streaming transpose buffer cap in MB
static int g_min_section_size = 1;
static bool g_min_section_explicit = false;
static int64_t g_max_pairs_cap = 0;   // 0 = uncapped; >0 = skip sweep steps exceeding cap
static double g_size_variance = 0.0;   // 0 = exact sizes; >0 = ±fraction (e.g. 0.3 = ±30%)

// Scale mode
static std::string g_sweep_param = "section_size";
static int g_steps = 10;
static int g_warmup = 1;
static int g_repeat = 3;
static std::string g_distribution = "random";
static int g_num_free_a = -1; // -1 = use default
static int g_num_free_b = -1;
static int g_num_contracted = -1;
static int g_num_batch = -1; // -1 = use default (0)
static bool g_section_size_explicit = false;
static bool g_density_explicit = false;
static bool g_distribution_explicit = false;

// Tier isolation and verification (scale mode)
static std::string g_tier = "";      // "large", "medium", "small", or "" (no constraint)
static bool g_no_verify = false;     // skip verification in scale mode

// Per-role section size overrides (scale mode)
static int g_free_a_size = 0;       // 0 = use default; >0 = pin free_A sections
static int g_free_b_size = 0;       // 0 = use default; >0 = pin free_B sections
static int g_contracted_size = 0;   // 0 = use default; >0 = pin contracted sections
static bool g_no_shuffle = false;   // disable label shuffling
static int g_vol_max_tiles = 0;    // 0 = auto; >0 = override medium tile limit

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------
static void parse_args(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> const char * {
      if (i + 1 < argc)
        return argv[++i];
      std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
      std::exit(1);
    };

    if (arg == "--fuzz") {
      g_scale_mode = false;
    } else if (arg == "--scale") {
      g_scale_mode = true;
    } else if (arg == "--seed") {
      g_seed = std::stoull(next());
    } else if (arg == "--count") {
      g_count = std::stoi(next());
    } else if (arg == "--max-rank") {
      g_max_rank = std::stoi(next());
    } else if (arg == "--max-sections") {
      g_max_sections = std::stoi(next());
    } else if (arg == "--max-section-size") {
      g_max_section_size = std::stoi(next());
      g_section_size_explicit = true;
    } else if (arg == "--section-size") {
      int sz = std::stoi(next());
      g_min_section_size = sz;
      g_min_section_explicit = true;
      g_max_section_size = sz;
      g_section_size_explicit = true;
    } else if (arg == "--min-section-size") {
      g_min_section_size = std::stoi(next());
      g_min_section_explicit = true;
    } else if (arg == "--density") {
      g_density = std::stod(next());
      g_density_explicit = true;
    } else if (arg == "--distribution") {
      g_distribution = next();
      g_distribution_explicit = true;
    } else if (arg == "--num-free-a") {
      g_num_free_a = std::stoi(next());
    } else if (arg == "--num-free-b") {
      g_num_free_b = std::stoi(next());
    } else if (arg == "--num-contracted") {
      g_num_contracted = std::stoi(next());
    } else if (arg == "--num-batch") {
      g_num_batch = std::stoi(next());
    } else if (arg == "--dtype") {
      std::string dt = next();
      g_f64 = (dt == "f64" || dt == "float64");
    } else if (arg == "--stop-on-fail") {
      g_stop_on_fail = true;
    } else if (arg == "--threshold") {
      g_mn_threshold = std::stoll(next());
    } else if (arg == "--verbose") {
      g_verbose = true;
    } else if (arg == "--mem-limit") {
      g_mem_limit_bytes = std::stoll(next()) * 1024LL * 1024;
    } else if (arg == "--streaming") {
      g_streaming_cap_mb = std::stoll(next());
    } else if (arg == "--max-pairs") {
      g_max_pairs_cap = std::stoll(next());
    } else if (arg == "--size-variance") {
      g_size_variance = std::stod(next());
    } else if (arg == "--tier") {
      g_tier = next();
      if (g_tier != "small" && g_tier != "medium" && g_tier != "large") {
        std::fprintf(stderr, "Unknown tier: %s (use: small, medium, large)\n",
                     g_tier.c_str());
        std::exit(1);
      }
    } else if (arg == "--no-verify") {
      g_no_verify = true;
    } else if (arg == "--free-a-size") {
      g_free_a_size = std::stoi(next());
    } else if (arg == "--free-b-size") {
      g_free_b_size = std::stoi(next());
    } else if (arg == "--contracted-size") {
      g_contracted_size = std::stoi(next());
    } else if (arg == "--no-shuffle") {
      g_no_shuffle = true;
    } else if (arg == "--vol-max-tiles") {
      g_vol_max_tiles = std::stoi(next());
    } else if (arg == "--sweep") {
      g_sweep_param = next();
    } else if (arg == "--steps") {
      g_steps = std::stoi(next());
    } else if (arg == "--warmup") {
      g_warmup = std::stoi(next());
    } else if (arg == "--repeat") {
      g_repeat = std::stoi(next());
    } else if (arg == "--help" || arg == "-h") {
      std::printf(
          "Usage: fuzz_pipeline [--fuzz|--scale] [options]\n\n"
          "Modes:\n"
          "  --fuzz                 Random fuzzing with correctness "
          "verification (default)\n"
          "  --scale                Scaling study: sweep one parameter, report "
          "timing\n\n"
          "Common options:\n"
          "  --seed N               RNG seed (0=time-based for fuzz, 42 for "
          "scale)\n"
          "  --density F            Block density [0,1] (default: 0.3 fuzz, "
          "0.5 scale)\n"
          "  --threshold N          M*N threshold for CUTLASS (default: 4096)\n"
          "  --max-section-size N   Max block section size (default: 64)\n"
          "  --min-section-size N   Min block section size (default: 1)\n"
          "  --section-size N       Fix all block sections to this size\n"
          "  --max-sections N       Max sections per dimension (default: 5)\n"
          "  --distribution D       Section size distribution (default: "
          "random)\n"
          "                         uniform   - all sections same size per "
          "dimension\n"
          "                         random    - each section independently "
          "random\n"
          "                         mixed     - bimodal: alternating small and "
          "large\n"
          "                         power_law - many small, few large (cubic)\n"
          "  --dtype f32|f64        Data type (default: f32)\n"
          "  --verbose              Print details for each problem\n"
          "  --mem-limit N          Memory limit in MB (default: 2048)\n"
          "  --streaming N          Streaming transpose buffer cap in MB (0=disabled)\n"
          "  --max-pairs N          Skip sweep steps where pairs > N (0=uncapped)\n"
          "  --size-variance F      Section size variance (0=exact, 0.3=±30%%, default: 0)\n\n"
          "Fuzz options:\n"
          "  --count N              Number of iterations (0=infinite, default: "
          "1000)\n"
          "  --max-rank N           Maximum tensor rank (default: 5)\n"
          "  --stop-on-fail         Stop on first failure\n\n"
          "Scale options:\n"
          "  --sweep PARAM[,PARAM]  Parameter(s) to sweep (default: "
          "section_size)\n"
          "                         block_size   - block size (alias: "
          "section_size)\n"
          "                         num_blocks   - sections per dim "
          "(log-spaced 2..max)\n"
          "                         density      - block density "
          "(0.05..1.0)\n"
          "                         rank         - tensor rank "
          "(2..max_rank)\n"
          "                         all          - sweep all four axes "
          "sequentially\n"
          "  --tier TIER            Isolate one dispatch tier (large, medium, "
          "small)\n"
          "                         large  - section_size 64..256 (CUTLASS "
          "grouped GEMM)\n"
          "                         medium - volume GEMM "
          "(ceil(M/2)*ceil(N/2)<=256)\n"
          "                         small  - section_size 1..3 (warp GEMM)\n"
          "  --no-verify            Skip correctness verification (default: "
          "verify)\n"
          "  --steps N              Number of sweep points (default: 10)\n"
          "  --warmup N             Warmup iterations (default: 1)\n"
          "  --repeat N             Repeat iterations (default: 3)\n"
          "  --num-free-a N         Free indices in A (default: 1)\n"
          "  --num-free-b N         Free indices in B (default: 1)\n"
          "  --num-contracted N     Contracted indices (default: 1)\n"
          "  --num-batch N          Batch indices (default: 0)\n"
          "  --free-a-size S        Pin free_A section sizes to S\n"
          "  --free-b-size S        Pin free_B section sizes to S\n"
          "  --contracted-size S    Pin contracted section sizes to S\n"
          "  --no-shuffle           Disable label shuffling (canonical layout)\n"
          "  --vol-max-tiles N      Override medium tier tile limit (default: auto)\n\n"
          "Examples:\n"
          "  fuzz_pipeline --scale --tier large --sweep block_size --steps 8\n"
          "  fuzz_pipeline --scale --tier small --sweep num_blocks --steps 8\n"
          "  fuzz_pipeline --scale --tier medium --sweep all --steps 6\n"
          "  fuzz_pipeline --scale --sweep block_size,density --steps 8\n"
          "  fuzz_pipeline --scale --no-verify --sweep block_size --steps 10\n"
          "  fuzz_pipeline --fuzz --count 100 --seed 42 --verbose\n");
      std::exit(0);
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
      std::fprintf(stderr, "Run with --help for usage\n");
      std::exit(1);
    }
  }
}

// ---------------------------------------------------------------------------
// Timing helper: warmup + repeat, return median
// ---------------------------------------------------------------------------
template <typename Fn>
static double bench_median(int warmup, int repeat, Fn &&fn) {
  for (int i = 0; i < warmup; ++i)
    fn();
  std::vector<double> times(static_cast<size_t>(repeat));
  for (int i = 0; i < repeat; ++i) {
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    times[static_cast<size_t>(i)] =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  std::sort(times.begin(), times.end());
  return times[static_cast<size_t>(repeat / 2)];
}

// ---------------------------------------------------------------------------
// Run one fuzz problem, return: 0=pass, 1=fail, 2=skip, 3=error
// ---------------------------------------------------------------------------
template <typename T>
static int run_one_fuzz(int iter, uint64_t seed, bool verbose) {
  // Randomize params from meta-RNG
  std::mt19937 meta_rng(seed);
  auto params = randomize_params(meta_rng, g_max_rank, g_max_sections,
                                 g_max_section_size, g_density, seed);

  // Batch dims add overhead without stressing new code paths — keep rare (~5%)
  if (params.num_batch > 0 && (seed % 20 != 0))
    params.num_batch = 0;

  DataType dtype =
      std::is_same<T, double>::value ? DataType::FLOAT64 : DataType::FLOAT32;
  RandomContraction rc;
  try {
    rc = generate_random_contraction(params, dtype, g_mem_limit_bytes);
  } catch (const std::exception &e) {
    if (verbose)
      std::printf("[%d] seed=%lu → SKIP (gen: %s)\n", iter, seed, e.what());
    return 2;
  }

  // Parse einsum and prepare inputs
  auto expr = parse_einsum(rc.einsum_str);
  std::vector<std::vector<int64_t>> shapes = {
      block_utils::compute_full_shape(rc.A),
      block_utils::compute_full_shape(rc.B)};
  validate_einsum(expr, shapes);

  EinsumIndexSets index_sets;
  create_einsum_index_sets(expr, index_sets);

  auto A_gpu = to_gpu_input<T>(rc.A);
  auto B_gpu = to_gpu_input<T>(rc.B);

  ContractionConfig config = build_contraction_config(
      index_sets, A_gpu.labels, B_gpu.labels, expr.output_labels);

  int A_blk = static_cast<int>(rc.A.block_coords.size());
  int B_blk = static_cast<int>(rc.B.block_coords.size());

  // Step 1: Run our pipeline
  GeneralSparseResult<T> pipeline_result{};
  PipelineTimingBreakdown timing{};
  try {
    DispatchConfig dc;
    dc.mn_threshold = g_mn_threshold;
    dc.transpose_buf_cap = (g_streaming_cap_mb * 1024LL * 1024) / static_cast<int64_t>(sizeof(T));
    dc.vol_max_tiles = g_vol_max_tiles;
    auto A_dev = copy_sparse_to_device(A_gpu);
    auto B_dev = copy_sparse_to_device(B_gpu);
    pipeline_result = execute_general_sparse_pipeline(A_gpu, B_gpu, A_dev, B_dev,
                                                      index_sets, expr.output_labels,
                                                      dc, &timing);

    // Sync and check for CUDA errors from the pipeline
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::printf("[%d] seed=%lu einsum=%s A=%dblk B=%dblk → ERROR (pipeline "
                  "CUDA: %s)\n",
                  iter, seed, rc.einsum_str.c_str(), A_blk, B_blk,
                  cudaGetErrorString(err));
      cudaGetLastError(); // clear

      return 3;
    }
  } catch (const std::exception &e) {
    std::printf(
        "[%d] seed=%lu einsum=%s A=%dblk B=%dblk → ERROR (pipeline: %s)\n",
        iter, seed, rc.einsum_str.c_str(), A_blk, B_blk, e.what());
    cudaGetLastError();
    return 3;
  }

  // Step 2: Run cuTENSOR block-sparse reference
  CutensorBlockSparseResult<T> bs_ref;
  try {
    bs_ref = run_cutensor_block_sparse<T>(rc.A, rc.B, expr);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::printf("[%d] seed=%lu einsum=%s A=%dblk B=%dblk → UNCHECKED "
                  "(cuTENSOR CUDA: %s, %.2fms)\n",
                  iter, seed, rc.einsum_str.c_str(), A_blk, B_blk,
                  cudaGetErrorString(err), timing.total_ms);
      cudaGetLastError();
      return 2;
    }
  } catch (const std::exception &e) {
    std::printf("[%d] seed=%lu einsum=%s A=%dblk B=%dblk → UNCHECKED (cuTENSOR "
                "ref: %s, %.2fms)\n",
                iter, seed, rc.einsum_str.c_str(), A_blk, B_blk, e.what(),
                timing.total_ms);
    cudaGetLastError();
    return 2;
  }

  // Step 3: Verify pipeline output against cuTENSOR block-sparse reference
  double abs_tol = std::is_same<T, double>::value ? 1e-10 : 1e-3;
  double rel_tol = std::is_same<T, double>::value ? 1e-8 : 1e-3;
  auto vr = verify_against_block_sparse(pipeline_result, A_gpu, B_gpu, config,
                                        bs_ref, abs_tol, rel_tol);
  bool pass = vr.passed;

  if (pass) {
    std::printf("[%d] seed=%lu einsum=%s A=%dblk B=%dblk → PASSED (rel=%.1e, "
                "abs=%.1e, %.2fms)\n",
                iter, seed, rc.einsum_str.c_str(), A_blk, B_blk, vr.max_rel,
                vr.max_abs, timing.total_ms);
    return 0;
  } else {
    std::printf("[%d] seed=%lu einsum=%s A=%dblk B=%dblk → FAILED (rel=%.1e, "
                "abs=%.1e)\n",
                iter, seed, rc.einsum_str.c_str(), A_blk, B_blk, vr.max_rel,
                vr.max_abs);
    if (verbose) {
      std::printf("  Params: free_A=%d free_B=%d contracted=%d batch=%d\n",
                  params.num_free_A, params.num_free_B, params.num_contracted,
                  params.num_batch);
      std::printf("  sections=%d-%d, section_size=%d-%d, density=%.2f\n",
                  params.min_sections, params.max_sections,
                  params.min_section_size, params.max_section_size,
                  params.density);
    }
    return 1;
  }
}

// ---------------------------------------------------------------------------
// Fuzz mode main loop
// ---------------------------------------------------------------------------
template <typename T> static void run_fuzz_mode() {
  uint64_t base_seed = g_seed;
  if (base_seed == 0) {
    base_seed = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
  }

  std::printf("Fuzzing with base_seed=%lu, count=%d, max_rank=%d, "
              "max_sections=%d, max_section_size=%d, density=%.2f\n",
              base_seed, g_count, g_max_rank, g_max_sections,
              g_max_section_size, g_density);
  std::printf("Memory limit: %.0f MB, threshold: %ld",
              static_cast<double>(g_mem_limit_bytes) / (1024.0 * 1024.0),
              static_cast<long>(g_mn_threshold));
  if (g_streaming_cap_mb > 0)
    std::printf(", streaming: %ld MB", static_cast<long>(g_streaming_cap_mb));
  std::printf("\n\n");

  int passed = 0, failed = 0, skipped = 0, errors = 0;

  for (int i = 0; i < g_count || g_count == 0; ++i) {
    uint64_t seed = base_seed + static_cast<uint64_t>(i);
    int result = run_one_fuzz<T>(i, seed, g_verbose);

    switch (result) {
    case 0:
      ++passed;
      break;
    case 1:
      ++failed;
      break;
    case 2:
      ++skipped;
      break;
    case 3:
      ++errors;
      break;
    }

    if (result == 1 && g_stop_on_fail) {
      std::printf("\nStopping on first failure. Reproduce with: "
                  "fuzz_pipeline --fuzz --seed %lu --count 1 --verbose\n",
                  seed);
      break;
    }

    // Check for sticky CUDA errors that would doom all subsequent iterations
    if (result == 3) {
      cudaError_t err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        cudaGetLastError();
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
          std::printf(
              "\nFatal: device in unrecoverable error state, stopping.\n");
          break;
        }
      }
    }
  }

  std::printf("\nSummary: %d passed, %d failed, %d skipped, %d errors "
              "(seeds %lu–%lu)\n",
              passed, failed, skipped, errors, base_seed,
              base_seed + static_cast<uint64_t>(g_count > 0 ? g_count - 1 : 0));
}

// ---------------------------------------------------------------------------
// Parse distribution string → enum
// ---------------------------------------------------------------------------
static SectionDistribution parse_distribution(const std::string &s) {
  if (s == "uniform")
    return SectionDistribution::UNIFORM;
  if (s == "random")
    return SectionDistribution::RANDOM;
  if (s == "mixed")
    return SectionDistribution::MIXED;
  if (s == "power_law")
    return SectionDistribution::POWER_LAW;
  std::fprintf(
      stderr,
      "Unknown distribution: %s (use: uniform, random, mixed, power_law)\n",
      s.c_str());
  std::exit(1);
}

// ---------------------------------------------------------------------------
// Tier section-size bounds
// ---------------------------------------------------------------------------
static void tier_section_bounds(int &lo, int &hi) {
  if (g_tier == "small") {
    // Warp tier: min(M,N) < 4, max(M,N) ≤ 1024.
    // Global lo/hi are for the "large" dims (free_B + contracted).
    // The "small" dim (free_A) is overridden per-role in make_scale_base_params.
    lo = 4;
    hi = g_section_size_explicit ? g_max_section_size : 64;
  } else if (g_tier == "medium") {
    // Volume GEMM requires ceil(M/2)*ceil(N/2) ≤ 256 where M = sec^nfA, N = sec^nfB.
    // Also M*K+N*K ≤ vol_threshold (8192 for f32, 4096 for f64).
    int nfa = (g_num_free_a > 0) ? g_num_free_a : 1;
    int nfb = (g_num_free_b > 0) ? g_num_free_b : 1;
    int nc  = (g_num_contracted > 0) ? g_num_contracted : 1;
    int vol_limit = g_f64 ? 4096 : 8192;
    // Start high, walk down until both constraints are met.
    hi = g_section_size_explicit ? g_max_section_size : 64;
    while (hi > 1) {
      int64_t M = 1, N = 1, K = 1;
      for (int i = 0; i < nfa; ++i) M *= hi;
      for (int i = 0; i < nfb; ++i) N *= hi;
      for (int i = 0; i < nc; ++i)  K *= hi;
      int tiles = ((int(M) + 1) / 2) * ((int(N) + 1) / 2);
      if (tiles <= 256 && (M * K + N * K) <= vol_limit) break;
      --hi;
    }
    lo = std::max(1, hi / 4);
  } else if (g_tier == "large") {
    lo = 64;
    hi = g_section_size_explicit ? g_max_section_size : 256;
  } else {
    lo = 4;
    hi = g_max_section_size;
  }

  // CLI overrides take precedence over tier defaults
  if (g_min_section_explicit)
    lo = g_min_section_size;
}

// ---------------------------------------------------------------------------
// Build base params for a scale sweep point (tier + CLI overrides applied)
// ---------------------------------------------------------------------------
static RandomContractionParams make_scale_base_params(uint64_t seed) {
  RandomContractionParams p;
  p.seed = seed;
  p.num_free_A = (g_num_free_a > 0) ? g_num_free_a : 1;
  p.num_free_B = (g_num_free_b > 0) ? g_num_free_b : 1;
  p.num_contracted = (g_num_contracted > 0) ? g_num_contracted : 1;
  p.num_batch = (g_num_batch >= 0) ? g_num_batch : 0;
  // Use --max-sections as the fixed grid size for scale mode.
  // num_blocks sweep overrides this per-step; other sweeps keep it.
  p.min_sections = g_max_sections;
  p.max_sections = g_max_sections;
  p.density = g_density_explicit ? g_density : 0.5;
  p.min_nnz = 1;
  p.shuffle_labels = true;
  p.distribution = parse_distribution(g_distribution);

  int lo, hi;
  tier_section_bounds(lo, hi);
  p.min_section_size = lo;
  p.max_section_size = hi;
  // When tier is set, default to uniform sections (1 shape) for consistency.
  // But if --distribution was explicitly specified, respect it for shape diversity.
  if (!g_tier.empty() && !g_distribution_explicit)
    p.uniform_sections = true;
  else if (!g_tier.empty() && g_distribution == "uniform")
    p.uniform_sections = true;

  // Small tier: free_A gets skinny, free_B + contracted get large.
  // This produces warp-tier shapes like M=3, N=64, K=32.
  if (g_tier == "small") {
    p.small_free_A_lo = 3;
    p.small_free_A_hi = 3;
  }

  // CLI per-role overrides (take precedence over tier defaults)
  if (g_free_a_size > 0) { p.small_free_A_lo = p.small_free_A_hi = g_free_a_size; }
  if (g_free_b_size > 0) { p.small_free_B_lo = p.small_free_B_hi = g_free_b_size; }
  if (g_contracted_size > 0) { p.contracted_lo = p.contracted_hi = g_contracted_size; }
  if (g_no_shuffle) { p.shuffle_labels = false; }

  // Apply size variance: expand section size range ±fraction around midpoint.
  // E.g. variance=0.3 with sec=20 → range [14, 26].
  if (g_size_variance > 0) {
    int mid = (p.min_section_size + p.max_section_size) / 2;
    int delta = std::max(1, static_cast<int>(std::round(mid * g_size_variance)));
    p.min_section_size = std::max(1, mid - delta);
    p.max_section_size = mid + delta;
    p.uniform_sections = false;
    if (p.distribution == SectionDistribution::UNIFORM)
      p.distribution = SectionDistribution::RANDOM;
  }

  return p;
}

// ---------------------------------------------------------------------------
// Generate sweep values for a parameter
// ---------------------------------------------------------------------------
static std::vector<double> make_sweep_values(const std::string &param) {
  std::vector<double> vals(static_cast<size_t>(g_steps));

  if (param == "block_size" || param == "section_size") {
    int lo, hi;
    tier_section_bounds(lo, hi);
    for (int i = 0; i < g_steps; ++i) {
      double t = static_cast<double>(i) / std::max(1, g_steps - 1);
      vals[static_cast<size_t>(i)] = std::round(
          static_cast<double>(lo) *
          std::pow(static_cast<double>(hi) / static_cast<double>(lo), t));
    }
  } else if (param == "num_blocks") {
    double lo = 2.0, hi = static_cast<double>(g_max_sections);
    for (int i = 0; i < g_steps; ++i) {
      double t = static_cast<double>(i) / std::max(1, g_steps - 1);
      vals[static_cast<size_t>(i)] = std::round(lo * std::pow(hi / lo, t));
    }
  } else if (param == "density" || param == "nnz") {
    for (int i = 0; i < g_steps; ++i) {
      vals[static_cast<size_t>(i)] =
          0.05 + 0.95 * static_cast<double>(i) / std::max(1, g_steps - 1);
    }
  } else if (param == "rank") {
    for (int i = 0; i < g_steps; ++i) {
      vals[static_cast<size_t>(i)] =
          2.0 + static_cast<double>(i) *
                    std::max(1.0, (static_cast<double>(g_max_rank) - 2.0) /
                                      std::max(1, g_steps - 1));
    }
  } else {
    std::fprintf(stderr, "Unknown sweep param: %s\n", param.c_str());
    std::fprintf(
        stderr,
        "Available: block_size, section_size, num_blocks, density, rank, all\n");
    std::exit(1);
  }

  // Deduplicate consecutive equal values (important for small ranges like
  // tier=small with many steps)
  std::vector<double> deduped;
  for (double v : vals) {
    if (deduped.empty() || v != deduped.back())
      deduped.push_back(v);
  }
  return deduped;
}

// ---------------------------------------------------------------------------
// Apply a sweep value override to base params
// ---------------------------------------------------------------------------
static void apply_sweep_override(RandomContractionParams &params,
                                 const std::string &param, double val) {
  if (param == "block_size" || param == "section_size") {
    int sz = static_cast<int>(val);
    if (g_size_variance > 0) {
      int delta = std::max(1, static_cast<int>(std::round(sz * g_size_variance)));
      params.min_section_size = std::max(1, sz - delta);
      params.max_section_size = sz + delta;
      params.uniform_sections = false;
    } else {
      params.min_section_size = sz;
      params.max_section_size = sz;
      params.uniform_sections = true;
    }
  } else if (param == "density" || param == "nnz") {
    params.density = val;
  } else if (param == "rank") {
    int rank = static_cast<int>(val);
    params.num_contracted = 1;
    int free_total = std::max(2, rank - 1);
    params.num_free_A = free_total / 2;
    params.num_free_B = free_total - params.num_free_A;
  } else if (param == "num_blocks") {
    int sections = static_cast<int>(val);
    params.min_sections = sections;
    params.max_sections = sections;
    // Use small fixed blocks when no tier and no explicit size
    if (!g_section_size_explicit && g_tier.empty()) {
      params.min_section_size = 4;
      params.max_section_size = 4;
      params.uniform_sections = true;
    }
  }
}

// ---------------------------------------------------------------------------
// Parse sweep spec: "block_size", "block_size,density", "all"
// ---------------------------------------------------------------------------
static std::vector<std::string> parse_sweep_spec(const std::string &spec) {
  if (spec == "all")
    return {"block_size", "num_blocks", "density", "rank"};

  std::vector<std::string> result;
  size_t start = 0;
  while (start < spec.size()) {
    size_t comma = spec.find(',', start);
    if (comma == std::string::npos) {
      result.push_back(spec.substr(start));
      break;
    }
    result.push_back(spec.substr(start, comma - start));
    start = comma + 1;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Compute representative M, N, K from block shapes
// ---------------------------------------------------------------------------
// For the first block (section 0 in each dim), compute:
//   M = product of free_A section sizes
//   N = product of free_B section sizes
//   K = product of contracted section sizes
// Returns {M, N, K}.  For random shapes this is the "section-0" representative.
static std::array<int64_t, 3> compute_representative_mnk(
    const BlockSparseTensor &A,
    const BlockSparseTensor &B,
    const EinsumIndexSets &index_sets) {

  // Section-0 size for a given index label (look in A first, then B)
  auto sec0 = [&](char label) -> int64_t {
    auto it = A.block_sizes.find(label);
    if (it != A.block_sizes.end() && !it->second.empty())
      return it->second[0];
    it = B.block_sizes.find(label);
    if (it != B.block_sizes.end() && !it->second.empty())
      return it->second[0];
    return 1;
  };

  int64_t M = 1, N = 1, K = 1;
  for (char c : index_sets.free_indices_A) M *= sec0(c);
  for (char c : index_sets.free_indices_B) N *= sec0(c);
  for (char c : index_sets.contracted_indices) K *= sec0(c);
  return {M, N, K};
}

// ---------------------------------------------------------------------------
// Run one sweep axis
// ---------------------------------------------------------------------------
template <typename T>
static void run_single_sweep(const std::string &sweep_name,
                             uint64_t base_seed, DataType dtype) {
  auto values = make_sweep_values(sweep_name);
  std::string tier_label = g_tier.empty() ? "all" : g_tier;

  std::printf(
      "\n=== Sweep: %s (tier=%s, seed=%lu, warmup=%d, repeat=%d, dist=%s) "
      "===\n",
      sweep_name.c_str(), tier_label.c_str(), base_seed, g_warmup, g_repeat,
      g_distribution.c_str());

  if (g_no_verify)
    std::printf("%10s %8s %8s %8s %6s %6s %6s %12s %10s %10s %10s %10s %8s %8s\n",
                "param", "M", "N", "K",
                "A_blk", "B_blk", "pairs", "L/M/S", "setup_ms", "exec_ms",
                "ct_setup", "ct_kern", "exec_su", "total_su");
  else
    std::printf("%10s %8s %8s %8s %6s %6s %6s %12s %10s %10s %10s %10s %8s %8s %6s\n",
                "param", "M", "N", "K",
                "A_blk", "B_blk", "pairs", "L/M/S", "setup_ms", "exec_ms",
                "ct_setup", "ct_kern", "exec_su", "total_su", "status");

  std::printf("--------------------------------------------------------------"
              "--------------------------------------------------------------"
              "-----------%s\n",
              g_no_verify ? "" : "-------");

  int verified_pass = 0, verified_fail = 0, verified_skip = 0;

  for (size_t step = 0; step < values.size(); ++step) {
    double val = values[step];

    auto params = make_scale_base_params(base_seed);
    apply_sweep_override(params, sweep_name, val);

    RandomContraction rc;
    try {
      rc = generate_random_contraction(params, dtype, g_mem_limit_bytes);
    } catch (const std::exception &e) {
      std::printf("%10.1f   SKIP (gen: %s)\n", val, e.what());
      ++verified_skip;
      continue;
    }

    try {
      auto expr = parse_einsum(rc.einsum_str);
      std::vector<std::vector<int64_t>> shapes = {
          block_utils::compute_full_shape(rc.A),
          block_utils::compute_full_shape(rc.B)};
      validate_einsum(expr, shapes);

      EinsumIndexSets index_sets;
      create_einsum_index_sets(expr, index_sets);

      auto [rep_M, rep_N, rep_K] = compute_representative_mnk(rc.A, rc.B, index_sets);

      auto A_gpu = to_gpu_input<T>(rc.A);
      auto B_gpu = to_gpu_input<T>(rc.B);

      // Build contraction config early (needed for pair count check)
      ContractionConfig config = build_contraction_config(
          index_sets, A_gpu.labels, B_gpu.labels, expr.output_labels);

      // Early pair count check (host-side, before GPU allocation)
      if (g_max_pairs_cap > 0) {
        auto check_pairs = find_matching_pairs(A_gpu, B_gpu, config);
        auto np = static_cast<int64_t>(check_pairs.size());
        if (np > g_max_pairs_cap) {
          std::printf("%10.1f   SKIP (pairs %ld > cap %ld)\n",
                      val, static_cast<long>(np),
                      static_cast<long>(g_max_pairs_cap));
          ++verified_skip;
          continue;
        }
      }

      DispatchConfig dc;
      dc.mn_threshold = g_mn_threshold;
      dc.transpose_buf_cap = (g_streaming_cap_mb * 1024LL * 1024) / static_cast<int64_t>(sizeof(T));
      dc.vol_max_tiles = g_vol_max_tiles;
      auto A_dev = copy_sparse_to_device(A_gpu);
      auto B_dev = copy_sparse_to_device(B_gpu);

      // Run pipeline once to get tier counts + result for verification
      int n_large = 0, n_medium = 0, n_small = 0;
      GeneralSparseResult<T> first_result{};
      {
        PipelineTimingBreakdown tb{};
        first_result = execute_general_sparse_pipeline(
            A_gpu, B_gpu, A_dev, B_dev, index_sets, expr.output_labels, dc,
            &tb);
        n_large = tb.large_gemm_problems;
        n_medium = tb.medium_descs;
        n_small = tb.small_descs;
      }
      int num_pairs = n_large + n_medium + n_small;

      // Verify correctness against cuTENSOR block-sparse reference
      const char *status = "-";
      if (!g_no_verify) {
        try {
          auto bs_ref = run_cutensor_block_sparse<T>(rc.A, rc.B, expr);
          cudaError_t err = cudaDeviceSynchronize();
          if (err != cudaSuccess) {
            cudaGetLastError();
            status = "SKIP";
            ++verified_skip;
          } else {
            double abs_tol =
                std::is_same<T, double>::value ? 1e-10 : 1e-3;
            double rel_tol =
                std::is_same<T, double>::value ? 1e-8 : 1e-3;
            auto vr = verify_against_block_sparse(first_result, A_gpu, B_gpu,
                                                  config, bs_ref, abs_tol,
                                                  rel_tol);
            if (vr.passed) {
              status = "PASS";
              ++verified_pass;
            } else {
              status = "FAIL";
              ++verified_fail;
              if (g_verbose)
                std::printf("  VERIFY FAIL: max_rel=%.3e max_abs=%.3e\n",
                            vr.max_rel, vr.max_abs);
            }
          }
        } catch (const std::exception &) {
          status = "SKIP";
          ++verified_skip;
        }
      }

      // Pre-compute output block structure (excluded from timing).
      // Simulates iterative pipeline where C's sparsity pattern is known.
      auto bench_pairs = find_matching_pairs(A_gpu, B_gpu, config);
      OutputBlockMap output_blocks = compute_output_blocks(
          bench_pairs.data(), static_cast<int64_t>(bench_pairs.size()),
          A_gpu, B_gpu, config);
      auto ct_output_coords = output_blocks.to_output_order(
          config.internal_to_output);

      // Benchmark setup (plan + prepare, excluding alloc/free to match cuTENSOR)
      PipelineTimingBreakdown last_tb{};
      bench_median(g_warmup, g_repeat, [&]() {
        PipelinePlan<T> plan;
        last_tb = PipelineTimingBreakdown{};
        pipeline_setup(plan, A_gpu, B_gpu, A_dev, B_dev,
                       index_sets, expr.output_labels, dc, &last_tb,
                       &output_blocks);
      });
      double setup_ms = last_tb.plan_ms + last_tb.prepare_ms + last_tb.bind_ms;

      // Benchmark execute-only (setup once, re-execute)
      PipelinePlan<T> bench_plan;
      pipeline_setup(bench_plan, A_gpu, B_gpu, A_dev, B_dev,
                     index_sets, expr.output_labels, dc, nullptr,
                     &output_blocks);
      double exec_ms = bench_median(g_warmup, g_repeat, [&]() {
        pipeline_execute(bench_plan, A_gpu, B_gpu, A_dev, B_dev);
      });

      // Benchmark cuTENSOR block-sparse
      // Setup once for setup_ms, then bench contract-only (wall-clock median)
      auto last_ct = run_cutensor_block_sparse<T>(rc.A, rc.B, expr,
          nullptr, nullptr, &ct_output_coords);
      double ct_setup_ms = static_cast<double>(last_ct.setup_ms);
      double ct_exec_ms;
      {
        auto ct_prepared = prepare_cutensor_block_sparse<T>(
            rc.A, rc.B, expr, &ct_output_coords);
        ct_exec_ms = bench_median(g_warmup, g_repeat, [&]() {
          ct_prepared.run();
        });
      }

      double total_ms = setup_ms + exec_ms;
      double ct_total_ms = ct_setup_ms + ct_exec_ms;
      double speedup_exec = ct_exec_ms / std::max(0.001, exec_ms);
      double speedup_total = ct_total_ms / std::max(0.001, total_ms);

      char tier_buf[32];
      std::snprintf(tier_buf, sizeof(tier_buf), "%d/%d/%d", n_large, n_medium,
                    n_small);

      if (g_no_verify)
        std::printf("%10.1f %8ld %8ld %8ld %6d %6d %6d %12s %10.4f %10.4f %10.4f %10.4f %7.1fx %7.1fx\n",
                    val, static_cast<long>(rep_M), static_cast<long>(rep_N),
                    static_cast<long>(rep_K),
                    static_cast<int>(rc.A.block_coords.size()),
                    static_cast<int>(rc.B.block_coords.size()), num_pairs,
                    tier_buf, setup_ms, exec_ms, ct_setup_ms, ct_exec_ms,
                    speedup_exec, speedup_total);
      else
        std::printf("%10.1f %8ld %8ld %8ld %6d %6d %6d %12s %10.4f %10.4f %10.4f %10.4f %7.1fx %7.1fx %6s\n",
                    val, static_cast<long>(rep_M), static_cast<long>(rep_N),
                    static_cast<long>(rep_K),
                    static_cast<int>(rc.A.block_coords.size()),
                    static_cast<int>(rc.B.block_coords.size()), num_pairs,
                    tier_buf, setup_ms, exec_ms, ct_setup_ms, ct_exec_ms,
                    speedup_exec, speedup_total, status);

      if (g_verbose) {
        std::printf("           einsum=%s  plan=%.3f prepare=%.3f bind=%.3f(up=%.3f patch=%.3f strm=%.3f) alloc=%.3f | "
                    "preproc=%.3f accum_map=%.3f transpose=%.3f "
                    "contract=%.3f accumulate=%.3f ms"
                    "  med_shapes=%d"
                    "  ct_plan=%.3f ct_kern=%.3f\n",
                    rc.einsum_str.c_str(),
                    last_tb.plan_ms, last_tb.prepare_ms, last_tb.bind_ms,
                    last_tb.bind_upload_ms, last_tb.bind_patch_ms, last_tb.bind_stream_ms,
                    last_tb.allocate_ms,
                    last_tb.preprocess_ms, last_tb.accum_map_ms,
                    last_tb.transpose_ms, last_tb.contraction_ms,
                    last_tb.accumulation_ms, last_tb.medium_shapes,
                    ct_setup_ms, ct_exec_ms);
      }

    } catch (const std::exception &e) {
      std::printf("%10.1f   ERROR (%s)\n", val, e.what());
      cudaGetLastError();
      cudaDeviceSynchronize();
    }
  }

  if (!g_no_verify && (verified_pass + verified_fail + verified_skip) > 0) {
    std::printf("Verification: %d passed, %d failed, %d skipped\n",
                verified_pass, verified_fail, verified_skip);
  }
}

// ---------------------------------------------------------------------------
// Scale mode: dispatch sweep(s)
// ---------------------------------------------------------------------------
template <typename T> static void run_scale_mode() {
  uint64_t base_seed = g_seed;
  if (base_seed == 0)
    base_seed = 42;

  DataType dtype =
      std::is_same<T, double>::value ? DataType::FLOAT64 : DataType::FLOAT32;

  auto sweeps = parse_sweep_spec(g_sweep_param);
  for (const auto &s : sweeps) {
    run_single_sweep<T>(s, base_seed, dtype);
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  parse_args(argc, argv);

  if (g_scale_mode) {
    if (g_f64)
      run_scale_mode<double>();
    else
      run_scale_mode<float>();
  } else {
    if (g_f64)
      run_fuzz_mode<double>();
    else
      run_fuzz_mode<float>();
  }

  return 0;
}
