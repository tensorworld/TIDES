#pragma once

#include "../block_sparse_tensor.hpp"
#include "../block_utils.hpp"
#include "../einsum.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

namespace bsptc {

// Single block pair record for CSV output
struct BlockPairRecord {
    int64_t A_id;
    int64_t B_id;
    int64_t C_id;
    int64_t batch_size;
    int64_t free_A_size;
    int64_t free_B_size;
    int64_t contraction_size;
};

// Simulation result with statistics
struct SimulationResult {
    std::vector<BlockPairRecord> records;

    // Aggregate statistics
    int64_t num_A_blocks = 0;
    int64_t num_B_blocks = 0;
    int64_t num_pairs_checked = 0;
    int64_t num_matching_pairs = 0;
    int64_t num_unique_C_blocks = 0;

    // Write CSV to file
    void write_csv(const std::string& filename) const {
        std::ofstream f(filename);
        f << "A_id,B_id,C_id,batch_size,free_A_size,free_B_size,contraction_size\n";
        for (const auto& r : records) {
            f << r.A_id << ","
              << r.B_id << ","
              << r.C_id << ","
              << r.batch_size << ","
              << r.free_A_size << ","
              << r.free_B_size << ","
              << r.contraction_size << "\n";
        }
        f.close();
    }

    // Write CSV to stream
    void write_csv(std::ostream& os) const {
        os << "A_id,B_id,C_id,batch_size,free_A_size,free_B_size,contraction_size\n";
        for (const auto& r : records) {
            os << r.A_id << ","
               << r.B_id << ","
               << r.C_id << ","
               << r.batch_size << ","
               << r.free_A_size << ","
               << r.free_B_size << ","
               << r.contraction_size << "\n";
        }
    }

    void print_summary(std::ostream& os = std::cout) const {
        os << "=== Simulation Summary ===\n";
        os << "  A blocks:        " << num_A_blocks << "\n";
        os << "  B blocks:        " << num_B_blocks << "\n";
        os << "  Pairs checked:   " << num_pairs_checked << "\n";
        os << "  Matching pairs:  " << num_matching_pairs << "\n";
        os << "  Unique C blocks: " << num_unique_C_blocks << "\n";
        if (num_pairs_checked > 0) {
            double match_rate = 100.0 * static_cast<double>(num_matching_pairs) / static_cast<double>(num_pairs_checked);
            os << "  Match rate:      " << match_rate << "%\n";
        }
        if (num_unique_C_blocks > 0) {
            double avg_contention = static_cast<double>(num_matching_pairs) / static_cast<double>(num_unique_C_blocks);
            os << "  Avg C contention: " << avg_contention << " pairs/block\n";
        }
    }
};

// Group-level contraction simulator
// Works directly with BlockSparseTensor - no value access needed
class ContractionSimulator {
public:
    static SimulationResult simulate(
        const BlockSparseTensor& A,
        const BlockSparseTensor& B,
        const EinsumIndexSets& index_sets,
        const std::string& output_labels)
    {
        SimulationResult result;
        result.num_A_blocks = static_cast<int64_t>(A.block_coords.size());
        result.num_B_blocks = static_cast<int64_t>(B.block_coords.size());

        const int32_t A_rank = static_cast<int32_t>(A.meta.ndim);
        const int32_t B_rank = static_cast<int32_t>(B.meta.ndim);

        // Build index position maps
        std::map<char, int32_t> A_label_to_pos, B_label_to_pos;
        for (int32_t i = 0; i < A_rank; ++i) A_label_to_pos[A.meta.indices[i]] = i;
        for (int32_t i = 0; i < B_rank; ++i) B_label_to_pos[B.meta.indices[i]] = i;

        // Classify indices
        std::vector<char> contracted, batch, free_A, free_B;
        for (char c : index_sets.contracted_indices) contracted.push_back(c);
        for (char c : index_sets.batch_indices) batch.push_back(c);
        for (char c : index_sets.free_indices_A) free_A.push_back(c);
        for (char c : index_sets.free_indices_B) free_B.push_back(c);

        // Build output label to position map
        std::map<char, int32_t> out_label_to_pos;
        for (size_t i = 0; i < output_labels.size(); ++i) {
            out_label_to_pos[output_labels[i]] = static_cast<int32_t>(i);
        }
        int32_t out_rank = static_cast<int32_t>(output_labels.size());

        // Track unique output blocks and assign C_id
        std::map<std::vector<int32_t>, int64_t> output_coord_to_id;
        int64_t next_c_id = 0;

        // Helper to check if blocks match on given indices
        auto blocks_match = [&](const std::vector<int64_t>& A_coord,
                                const std::vector<int64_t>& B_coord,
                                const std::vector<char>& indices) {
            for (char c : indices) {
                int32_t A_pos = A_label_to_pos[c];
                int32_t B_pos = B_label_to_pos[c];
                if (A_coord[A_pos] != B_coord[B_pos]) return false;
            }
            return true;
        };

        // Convert block_coords sets to vectors for indexing
        std::vector<std::vector<int64_t>> A_coords(A.block_coords.begin(), A.block_coords.end());
        std::vector<std::vector<int64_t>> B_coords(B.block_coords.begin(), B.block_coords.end());

        // Process all block pairs
        for (int64_t a_blk = 0; a_blk < result.num_A_blocks; ++a_blk) {
            const auto& A_coord = A_coords[a_blk];
            auto A_dims = block_utils::block_shape_for_coord(A, A_coord);

            for (int64_t b_blk = 0; b_blk < result.num_B_blocks; ++b_blk) {
                result.num_pairs_checked++;

                const auto& B_coord = B_coords[b_blk];
                auto B_dims = block_utils::block_shape_for_coord(B, B_coord);

                // Check if blocks match on contracted and batch indices
                if (!blocks_match(A_coord, B_coord, contracted)) continue;
                if (!blocks_match(A_coord, B_coord, batch)) continue;

                result.num_matching_pairs++;

                // Compute output block coordinate
                std::vector<int32_t> out_coord(out_rank);
                for (char c : batch) {
                    out_coord[out_label_to_pos[c]] = static_cast<int32_t>(A_coord[A_label_to_pos[c]]);
                }
                for (char c : free_A) {
                    out_coord[out_label_to_pos[c]] = static_cast<int32_t>(A_coord[A_label_to_pos[c]]);
                }
                for (char c : free_B) {
                    out_coord[out_label_to_pos[c]] = static_cast<int32_t>(B_coord[B_label_to_pos[c]]);
                }

                // Assign C_id
                int64_t c_id;
                auto it = output_coord_to_id.find(out_coord);
                if (it == output_coord_to_id.end()) {
                    c_id = next_c_id++;
                    output_coord_to_id[out_coord] = c_id;
                } else {
                    c_id = it->second;
                }

                // Compute dimension products
                int64_t batch_size = 1;
                for (char c : batch) {
                    batch_size *= A_dims[A_label_to_pos[c]];
                }

                int64_t free_A_size = 1;
                for (char c : free_A) {
                    free_A_size *= A_dims[A_label_to_pos[c]];
                }

                int64_t free_B_size = 1;
                for (char c : free_B) {
                    free_B_size *= B_dims[B_label_to_pos[c]];
                }

                int64_t contraction_size = 1;
                for (char c : contracted) {
                    contraction_size *= A_dims[A_label_to_pos[c]];
                }

                // Record this pair
                BlockPairRecord rec;
                rec.A_id = a_blk;
                rec.B_id = b_blk;
                rec.C_id = c_id;
                rec.batch_size = batch_size;
                rec.free_A_size = free_A_size;
                rec.free_B_size = free_B_size;
                rec.contraction_size = contraction_size;
                result.records.push_back(rec);
            }
        }

        result.num_unique_C_blocks = static_cast<int64_t>(output_coord_to_id.size());

        return result;
    }
};

} // namespace bsptc
