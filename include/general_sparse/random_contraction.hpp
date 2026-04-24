#pragma once

#include "../block_sparse_tensor.hpp"
#include "../block_utils.hpp"
#include "../data_init_utils.hpp"
#include "../einsum.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace bsptc {

// ---------------------------------------------------------------------------
// Block section size distribution modes
// ---------------------------------------------------------------------------
enum class SectionDistribution {
    UNIFORM,    // All sections in a dimension have the same random size
    RANDOM,     // Each section independently random in [min, max] (default)
    MIXED,      // Bimodal: alternating small and large sections
    POWER_LAW   // Many small, few large (cubic distribution)
};

// ---------------------------------------------------------------------------
// Configurable parameter space for random contraction generation
// ---------------------------------------------------------------------------
struct RandomContractionParams {
    int num_free_A      = 1;      // Free indices in A (1–3)
    int num_free_B      = 1;      // Free indices in B (1–3)
    int num_contracted  = 1;      // Contracted indices (1–3)
    int num_batch       = 0;      // Batch indices (0–2)

    int min_sections    = 1;      // Min block sections per dim
    int max_sections    = 5;      // Max block sections per dim
    int min_section_size = 1;     // Min size of each section
    int max_section_size = 64;    // Max size of each section
    bool uniform_sections = false; // All sections same size? (overrides distribution to UNIFORM)
    SectionDistribution distribution = SectionDistribution::RANDOM;

    double density      = 0.3;    // Fraction of Cartesian-product blocks that are nnz
    int min_nnz         = 1;      // Floor on nnz per tensor

    bool shuffle_labels = true;   // Shuffle index ordering within tensors

    // Per-role section size overrides (0 = use global min/max_section_size).
    int small_free_A_lo = 0;    // Override min section size for free_A labels
    int small_free_A_hi = 0;    // Override max section size for free_A labels
    int small_free_B_lo = 0;    // Override min section size for free_B labels
    int small_free_B_hi = 0;    // Override max section size for free_B labels
    int contracted_lo   = 0;    // Override min section size for contracted labels
    int contracted_hi   = 0;    // Override max section size for contracted labels

    uint64_t seed       = 42;     // RNG seed
};

// ---------------------------------------------------------------------------
// Result of random contraction generation
// ---------------------------------------------------------------------------
struct RandomContraction {
    BlockSparseTensor A;
    BlockSparseTensor B;
    std::string einsum_str;        // e.g. "bdc,bce->de"
    RandomContractionParams params; // The params used (for reproducibility)
};

// ---------------------------------------------------------------------------
// Estimate peak GPU memory (bytes) for a random contraction problem.
// Conservative upper bound: dense output + 2x input values + temp buffer.
// ---------------------------------------------------------------------------
inline int64_t estimate_memory_bytes(const RandomContraction& rc) {
    auto A_shape = block_utils::compute_full_shape(rc.A);
    auto B_shape = block_utils::compute_full_shape(rc.B);

    // Parse einsum to figure out output shape
    auto expr = parse_einsum(rc.einsum_str);
    std::vector<std::vector<int64_t>> shapes = {A_shape, B_shape};
    validate_einsum(expr, shapes);

    // Output dense volume
    int64_t output_vol = 1;
    for (auto& s : expr.inputs[0].shape) { (void)s; }  // validate filled shapes
    // Compute output shape from output labels
    // Build label→size map from inputs
    std::unordered_map<char, int64_t> label_size;
    for (size_t i = 0; i < expr.inputs.size(); ++i) {
        for (size_t j = 0; j < expr.inputs[i].labels.size(); ++j) {
            label_size[expr.inputs[i].labels[j]] = expr.inputs[i].shape[j];
        }
    }
    for (char c : expr.output_labels) {
        output_vol *= label_size[c];
    }

    int64_t A_elems = static_cast<int64_t>(block_utils::total_elements(rc.A));
    int64_t B_elems = static_cast<int64_t>(block_utils::total_elements(rc.B));

    // Dense output (for reference) + sparse inputs on device + temp buffer ~= output_vol
    int64_t elem_size = 4;  // float
    return (output_vol + A_elems + B_elems + output_vol) * elem_size;
}

// ---------------------------------------------------------------------------
// Generate a random contraction problem
// ---------------------------------------------------------------------------
inline RandomContraction generate_random_contraction(
    const RandomContractionParams& params,
    DataType dtype = DataType::FLOAT32,
    int64_t mem_limit_bytes = 0)
{
    std::mt19937 rng(params.seed);

    // 1. Assign labels from pool a-z
    const std::string label_pool = "abcdefghijklmnopqrstuvwxyz";
    int total_labels = params.num_batch + params.num_free_A +
                       params.num_free_B + params.num_contracted;
    if (total_labels > 26) {
        throw std::runtime_error("Too many labels (max 26)");
    }

    int idx = 0;
    std::vector<char> batch_labels, free_A_labels, free_B_labels, contracted_labels;
    for (int i = 0; i < params.num_batch; ++i) batch_labels.push_back(label_pool[idx++]);
    for (int i = 0; i < params.num_free_A; ++i) free_A_labels.push_back(label_pool[idx++]);
    for (int i = 0; i < params.num_free_B; ++i) free_B_labels.push_back(label_pool[idx++]);
    for (int i = 0; i < params.num_contracted; ++i) contracted_labels.push_back(label_pool[idx++]);

    // 2. Generate section sizes per label
    //    lo_override/hi_override: per-role bounds (0 = use global)
    auto gen_sections = [&](int n_sections, int lo_override = 0, int hi_override = 0) -> std::vector<int64_t> {
        std::vector<int64_t> sizes(static_cast<size_t>(n_sections));
        auto dist_mode = params.distribution;
        if (params.uniform_sections) dist_mode = SectionDistribution::UNIFORM;

        int lo = (lo_override > 0) ? lo_override : params.min_section_size;
        int hi = (hi_override > 0) ? hi_override : params.max_section_size;

        switch (dist_mode) {
            case SectionDistribution::UNIFORM: {
                std::uniform_int_distribution<int> d(lo, hi);
                int64_t s = d(rng);
                for (auto& v : sizes) v = s;
                break;
            }
            case SectionDistribution::RANDOM: {
                std::uniform_int_distribution<int> d(lo, hi);
                for (auto& v : sizes) v = d(rng);
                break;
            }
            case SectionDistribution::MIXED: {
                // Bimodal: alternating small [lo, lo+range/3] and large [hi-range/3, hi]
                int range = hi - lo;
                int lo_hi = lo + std::max(1, range / 3);
                int hi_lo = hi - std::max(1, range / 3);
                if (lo_hi > hi) lo_hi = hi;
                if (hi_lo < lo) hi_lo = lo;
                std::uniform_int_distribution<int> lo_d(lo, lo_hi);
                std::uniform_int_distribution<int> hi_d(hi_lo, hi);
                for (int i = 0; i < n_sections; ++i) {
                    sizes[static_cast<size_t>(i)] = (i % 2 == 0) ? lo_d(rng) : hi_d(rng);
                }
                break;
            }
            case SectionDistribution::POWER_LAW: {
                // Cubic distribution: many small, few large
                std::uniform_real_distribution<double> u(0.0, 1.0);
                int range = hi - lo;
                for (auto& v : sizes) {
                    double x = u(rng);
                    v = lo + static_cast<int64_t>(std::round(range * x * x * x));
                }
                break;
            }
        }
        return sizes;
    };

    std::uniform_int_distribution<int> nsec_dist(params.min_sections, params.max_sections);

    // Shared labels: batch + contracted — same sections in A and B
    std::map<char, int> num_sections_map;
    std::map<char, std::vector<int64_t>> section_sizes_map;

    for (char c : batch_labels) {
        int ns = nsec_dist(rng);
        num_sections_map[c] = ns;
        section_sizes_map[c] = gen_sections(ns);
    }
    for (char c : contracted_labels) {
        int ns = nsec_dist(rng);
        num_sections_map[c] = ns;
        section_sizes_map[c] = gen_sections(ns, params.contracted_lo, params.contracted_hi);
    }
    // Free labels: independent, with optional per-role size overrides
    for (char c : free_A_labels) {
        int ns = nsec_dist(rng);
        num_sections_map[c] = ns;
        section_sizes_map[c] = gen_sections(ns, params.small_free_A_lo, params.small_free_A_hi);
    }
    for (char c : free_B_labels) {
        int ns = nsec_dist(rng);
        num_sections_map[c] = ns;
        section_sizes_map[c] = gen_sections(ns, params.small_free_B_lo, params.small_free_B_hi);
    }

    // 3. Build tensor index orderings
    // A = [batch, free_A, contracted], B = [batch, contracted, free_B]
    std::vector<char> A_labels, B_labels;
    A_labels.insert(A_labels.end(), batch_labels.begin(), batch_labels.end());
    A_labels.insert(A_labels.end(), free_A_labels.begin(), free_A_labels.end());
    A_labels.insert(A_labels.end(), contracted_labels.begin(), contracted_labels.end());

    B_labels.insert(B_labels.end(), batch_labels.begin(), batch_labels.end());
    B_labels.insert(B_labels.end(), contracted_labels.begin(), contracted_labels.end());
    B_labels.insert(B_labels.end(), free_B_labels.begin(), free_B_labels.end());

    // Output = [batch, free_A, free_B]
    std::vector<char> output_labels;
    output_labels.insert(output_labels.end(), batch_labels.begin(), batch_labels.end());
    output_labels.insert(output_labels.end(), free_A_labels.begin(), free_A_labels.end());
    output_labels.insert(output_labels.end(), free_B_labels.begin(), free_B_labels.end());

    // Shuffle A and B orderings if requested (produces interleaved/transposed layouts)
    if (params.shuffle_labels) {
        std::shuffle(A_labels.begin(), A_labels.end(), rng);
        std::shuffle(B_labels.begin(), B_labels.end(), rng);
    }

    // 4. Build BlockSparseTensor for A and B
    int64_t host_elems_total = 0;  // running count for early memory check
    auto build_tensor = [&](const std::string& name, const std::vector<char>& labels,
                            uint64_t sparsity_seed) -> BlockSparseTensor {
        int ndim = static_cast<int>(labels.size());
        std::vector<int> num_blocks(static_cast<size_t>(ndim));
        std::map<char, std::vector<int64_t>> block_sizes;

        for (int d = 0; d < ndim; ++d) {
            char c = labels[static_cast<size_t>(d)];
            num_blocks[static_cast<size_t>(d)] = num_sections_map[c];
            block_sizes[c] = section_sizes_map[c];
        }

        // Generate all Cartesian-product coordinates
        std::vector<std::vector<int64_t>> all_coords;
        int64_t total_blocks = 1;
        for (int d = 0; d < ndim; ++d) {
            total_blocks *= num_blocks[static_cast<size_t>(d)];
        }
        all_coords.reserve(static_cast<size_t>(total_blocks));

        // Generate Cartesian product iteratively
        all_coords.push_back({});
        for (int d = 0; d < ndim; ++d) {
            std::vector<std::vector<int64_t>> new_coords;
            new_coords.reserve(all_coords.size() * static_cast<size_t>(num_blocks[static_cast<size_t>(d)]));
            for (auto& coord : all_coords) {
                for (int s = 0; s < num_blocks[static_cast<size_t>(d)]; ++s) {
                    auto c = coord;
                    c.push_back(s);
                    new_coords.push_back(std::move(c));
                }
            }
            all_coords = std::move(new_coords);
        }

        // Select nnz blocks
        std::mt19937 sparsity_rng(sparsity_seed);
        std::shuffle(all_coords.begin(), all_coords.end(), sparsity_rng);

        int nnz = std::max(params.min_nnz,
                           static_cast<int>(std::ceil(params.density * static_cast<double>(total_blocks))));
        nnz = std::min(nnz, static_cast<int>(total_blocks));

        std::set<std::vector<int64_t>> block_coords(
            all_coords.begin(),
            all_coords.begin() + nnz);

        TensorMeta meta(name, static_cast<int16_t>(ndim), labels, dtype);
        BlockSparseTensor tensor(meta, num_blocks, block_sizes, block_coords);

        // Check host memory before allocating
        size_t total_elems = block_utils::total_elements(tensor);
        int64_t elem_bytes = (dtype == DataType::FLOAT64) ? 8 : 4;
        host_elems_total += static_cast<int64_t>(total_elems);
        if (mem_limit_bytes > 0 &&
            host_elems_total * elem_bytes > mem_limit_bytes) {
            throw std::runtime_error("host memory estimate " +
                std::to_string(host_elems_total * elem_bytes / (1024*1024)) +
                "MB exceeds limit");
        }

        tensor.host_data.set_data_type(dtype);
        tensor.host_data.allocate(total_elems);

        return tensor;
    };

    // Use different sub-seeds for A and B sparsity patterns
    std::uniform_int_distribution<uint64_t> seed_dist;
    uint64_t seed_A = seed_dist(rng);
    uint64_t seed_B = seed_dist(rng);

    BlockSparseTensor A = build_tensor("A", A_labels, seed_A);
    BlockSparseTensor B = build_tensor("B", B_labels, seed_B);

    // Fill with random data
    uint64_t data_seed_A = seed_dist(rng);
    uint64_t data_seed_B = seed_dist(rng);
    if (dtype == DataType::FLOAT32) {
        utils::init_random<float>(A, -1.0f, 1.0f, data_seed_A);
        utils::init_random<float>(B, -1.0f, 1.0f, data_seed_B);
    } else if (dtype == DataType::FLOAT64) {
        utils::init_random<double>(A, -1.0, 1.0, data_seed_A);
        utils::init_random<double>(B, -1.0, 1.0, data_seed_B);
    }

    // 5. Build einsum string
    std::string A_str(A_labels.begin(), A_labels.end());
    std::string B_str(B_labels.begin(), B_labels.end());
    std::string O_str(output_labels.begin(), output_labels.end());
    std::string einsum_str = A_str + "," + B_str + "->" + O_str;

    // 6. Validate
    auto expr = parse_einsum(einsum_str);
    std::vector<std::vector<int64_t>> shapes = {
        block_utils::compute_full_shape(A),
        block_utils::compute_full_shape(B)
    };
    validate_einsum(expr, shapes);

    RandomContraction rc;
    rc.A = std::move(A);
    rc.B = std::move(B);
    rc.einsum_str = std::move(einsum_str);
    rc.params = params;
    return rc;
}

// ---------------------------------------------------------------------------
// Randomize params themselves from a meta-RNG (for fuzz mode)
// ---------------------------------------------------------------------------
inline RandomContractionParams randomize_params(
    std::mt19937& meta_rng,
    int max_rank = 5,
    int max_sections = 5,
    int max_section_size = 64,
    double density = 0.3,
    uint64_t problem_seed = 42)
{
    RandomContractionParams p;
    p.seed = problem_seed;

    // Total rank budget: 2 to max_rank per tensor
    // At minimum: 1 free_A + 1 free_B + 1 contracted = rank 2 each
    // num_batch: 0-2, num_free_A: 1-3, num_free_B: 1-3, num_contracted: 1-3
    // constrained by total rank
    std::uniform_int_distribution<int> batch_dist(0, std::min(2, max_rank - 2));
    p.num_batch = batch_dist(meta_rng);

    int remaining = max_rank - p.num_batch;
    // Need at least 1 free_A + 1 contracted for A, 1 free_B + 1 contracted for B
    int max_free_A = std::min(3, remaining - 1);  // leave room for contracted
    std::uniform_int_distribution<int> fA_dist(1, std::max(1, max_free_A));
    p.num_free_A = fA_dist(meta_rng);

    remaining -= p.num_free_A;
    int max_free_B = std::min(3, remaining - 1);
    std::uniform_int_distribution<int> fB_dist(1, std::max(1, max_free_B));
    p.num_free_B = fB_dist(meta_rng);

    remaining -= p.num_free_B;
    int max_contracted = std::min(3, remaining);
    std::uniform_int_distribution<int> con_dist(1, std::max(1, max_contracted));
    p.num_contracted = con_dist(meta_rng);

    p.min_sections = 1;
    p.max_sections = max_sections;
    p.min_section_size = 1;
    p.max_section_size = max_section_size;

    std::uniform_int_distribution<int> uniform_dist(0, 3);
    p.uniform_sections = (uniform_dist(meta_rng) == 0);  // 25% chance

    p.density = density;
    p.min_nnz = 1;
    p.shuffle_labels = true;

    return p;
}

} // namespace bsptc
