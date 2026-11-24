#pragma once

#include "block_sparse_tensor.hpp"
#include "block_utils.hpp"
#include "data_init_utils.hpp"
#include "tensorsuite_parser.hpp"
#include "types.hpp"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <set>

// Parser for the custom .bs block-sparse format.
namespace bsptc {

// Parse a .bs text file into a BlockSparseTensor and seed its data buffer.
// Expected format (example):
//   # Tensor A (ab)
//   {2, 3}          <-- number of blocks along a,b
//   2               <-- nnz blocks
//   {4, 5, 6, 7, 8} <-- flattened block sizes per dimension (a sections then b)
//   0, 1            <-- block coordinate
//   1, 2
// Pseudocode:
//   read header to capture tensor name/indices
//   read block counts and nnz
//   read flattened block sizes then map per index
//   read nnz coordinates
//   validate counts and build tensor; allocate host buffer and fill random data
inline BlockSparseTensor parse_bs_tensor_file(const std::string& filename, DataType dtype, bool skip_values = false)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open .bs file: " + filename);
    }

    std::string line;

    std::string tensor_name;
    std::vector<char> tensor_indices;

    std::vector<int> num_blocks_per_dim;
    std::vector<std::int64_t> block_sizes_flat;

    std::set<std::vector<std::int64_t>> block_coords;

    std::int64_t nnz = -1;

    // -------------------------
    // Parsing state
    // -------------------------
    enum class State {
        Header,
        BlockCounts,
        NNZ,
        BlockSizes,
        Coordinates
    };

    State state = State::Header;
    while (std::getline(file, line)) {

        if (line.empty()) continue;

        // -------------------------
        // Header
        // -------------------------
        if (line[0] == '#') {
            if (line.find("# Tensor ") != std::string::npos) {
                size_t name_start = line.find("# Tensor ") + 9;
                size_t name_end = line.find(" (");
                if (name_end != std::string::npos)
                    tensor_name = line.substr(name_start,
                                              name_end - name_start);

                size_t l = line.find("(");
                size_t r = line.find(")");
                if (l != std::string::npos && r != std::string::npos) {
                    for (char c : line.substr(l + 1, r - l - 1)) {
                        if (c != ',' && c != ' ')
                            tensor_indices.push_back(c);
                    }
                }
            }
            continue;
        }

        // -------------------------
        // Block counts
        // -------------------------
        if (state == State::Header &&
            line.find('{') != std::string::npos)
        {
            std::string inside =
                line.substr(line.find('{') + 1,
                            line.find('}') -
                            line.find('{') - 1);

            std::istringstream ss(inside);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                num_blocks_per_dim.push_back(static_cast<int>(std::stoll(tok)));
            }

            state = State::NNZ;
            continue;
        }

        // -------------------------
        // NNZ
        // -------------------------
        if (state == State::NNZ) {
            nnz = std::stoll(line);
            if (nnz <= 0)
                throw std::runtime_error("Invalid nnz value");

            state = State::BlockSizes;
            continue;
        }

        // -------------------------
        // Block sizes (flattened list; split per dimension later)
        // -------------------------
        if (state == State::BlockSizes &&
            line.find('{') != std::string::npos)
        {
            std::string inside =
                line.substr(line.find('{') + 1,
                            line.find('}') -
                            line.find('{') - 1);

            std::istringstream ss(inside);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                block_sizes_flat.push_back(std::stoll(tok));
            }

            state = State::Coordinates;
            continue;
        }

        // -------------------------
        // Coordinates
        // -------------------------
        if (state == State::Coordinates) {
            std::istringstream ss(line);
            std::string tok;
            std::vector<std::int64_t> coord;
            while (std::getline(ss, tok, ',')) {
                coord.push_back(std::stoll(tok));
            }

            if (coord.size() != num_blocks_per_dim.size()) {
                throw std::runtime_error("Coordinate rank mismatch");
            }

            block_coords.insert(coord);

            if (static_cast<std::int64_t>(block_coords.size()) == nnz)
                break;
        }
    }

    if (static_cast<std::int64_t>(block_coords.size()) != nnz) {
        throw std::runtime_error("nnz mismatch: expected " +
                                 std::to_string(nnz) +
                                 ", got " +
                                 std::to_string(block_coords.size()));
    }

    // -------------------------
    // Build tensor
    // -------------------------
    if (tensor_indices.size() != num_blocks_per_dim.size()) {
        throw std::runtime_error("Rank mismatch between indices and block counts");
    }
    const std::size_t expected_sizes =
        std::accumulate(num_blocks_per_dim.begin(), num_blocks_per_dim.end(),
                        static_cast<std::size_t>(0));
    if (block_sizes_flat.size() != expected_sizes) {
        throw std::runtime_error("Block size count mismatch");
    }

    TensorMeta meta(
        tensor_name,
        static_cast<std::int16_t>(tensor_indices.size()),
        tensor_indices,
        dtype);

    std::map<char, std::vector<std::int64_t>> block_size_map;
    std::size_t offset = 0;
    for (std::size_t dim = 0; dim < tensor_indices.size(); ++dim) {
        const auto count = static_cast<std::size_t>(num_blocks_per_dim[dim]);
        block_size_map[tensor_indices[dim]] = std::vector<std::int64_t>(
            block_sizes_flat.begin() + offset,
            block_sizes_flat.begin() + offset + count);
        offset += count;
    }

    BlockSparseTensor tensor(meta, num_blocks_per_dim, block_size_map, block_coords);

    tensor.host_data.set_data_type(dtype);

    if (!skip_values) {
        const std::size_t total_block_elems = block_utils::total_elements(tensor);
        tensor.host_data.allocate(total_block_elems);
        // Seed host data with random values matching the tensor dtype.
        if (dtype == DataType::FLOAT64) {
            utils::init_random<double>(tensor, 0.0, 1.0);
        } else {
            utils::init_random<float>(tensor, 0.0f, 1.0f);
        }
    }
    return tensor;
}

inline bool is_tensorsuite_path(const std::string& path_str) {
    namespace fs = std::filesystem;
    const fs::path p(path_str);
    if (fs::is_directory(p)) return true;
    const std::string ext = p.extension().string();
    if (ext == ".tns" || ext == ".tnsb" || ext == ".json") return true;
    std::ifstream probe(path_str, std::ios::binary);
    if (probe.is_open()) {
        std::string first_line;
        if (std::getline(probe, first_line)) {
            if (first_line.rfind("%%TensorSuite-", 0) == 0) {
                return true;
            }
        }
    }
    return false;
}

// Unified loader supporting both TIDES .bs files and TensorSuite block-sparse
// bundles (bundle directory, *_metadata.json, .tns, or .tnsb).
// Note: .bs files embed authoritative mode labels in their header and are never
// relabeled; `labels` is only used for formats (like TensorSuite) that omit labels.
inline BlockSparseTensor parse_tensor_file(
    const std::string& filename,
    DataType dtype,
    const std::vector<char>& labels = {},
    bool skip_values = false)
{
    if (is_tensorsuite_path(filename)) {
        return parse_tensorsuite_tensor(filename, dtype, labels, skip_values);
    }
    return parse_bs_tensor_file(filename, dtype, skip_values);
}

inline std::vector<BlockSparseTensor> parse_tensor_files(
    const std::vector<std::string>& files,
    DataType dtype_,
    bool skip_values = false)
{
    std::vector<BlockSparseTensor> tensors;
    tensors.reserve(files.size());
    for (const auto& filename : files) {
        tensors.push_back(parse_tensor_file(filename, dtype_, {}, skip_values));
    }
    return tensors;
}

inline std::vector<BlockSparseTensor> parse_tensor_files(
    const std::vector<std::string>& files,
    DataType dtype_,
    const std::vector<std::vector<char>>& labels_per_tensor,
    bool skip_values = false)
{
    if (labels_per_tensor.size() != files.size()) {
        throw std::runtime_error("parse_tensor_files: labels count != files count");
    }
    std::vector<BlockSparseTensor> tensors;
    tensors.reserve(files.size());
    for (std::size_t i = 0; i < files.size(); ++i) {
        tensors.push_back(parse_tensor_file(files[i], dtype_, labels_per_tensor[i], skip_values));
    }
    return tensors;
}

} // namespace bsptc
