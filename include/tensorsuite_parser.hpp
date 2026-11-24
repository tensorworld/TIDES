#pragma once

#include "block_sparse_tensor.hpp"
#include "block_utils.hpp"
#include "data_init_utils.hpp"
#include "mini_json.hpp"
#include "types.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Parser for TensorSuite block-sparse bundles (*_metadata.json + .tns / .tnsb).
// Spec: https://tensorworld.github.io/TensorSuite/format.html
namespace bsptc {

struct TensorSuiteMetadata {
    std::string version;
    std::string name;
    std::string group;
    std::string id;
    std::string time;
    std::string source_type;
    std::string source;
    std::string source_url;

    std::string value_type = "float64";
    std::string value_domain = "general";
    bool values_provided = false;

    std::string index_type = "uint64";
    std::string endianness = "little";
    int index_base = 0;

    std::string sorted;
    std::string duplicates = "disallowed";
    std::string explicit_zeros;
    std::string pattern_symmetry;
    std::string numerical_symmetry;
    std::string sparsity_type;

    int order = 0;
    std::vector<std::int64_t> dimensions;
    std::int64_t nnz = 0;

    bool has_block_partitions = false;
    std::vector<int> block_partitions;
    bool has_nnz_block = false;
    std::int64_t nnz_block = 0;

    std::string file_text;
    std::string file_binary;
    std::string file_readme;
};

namespace detail {

inline std::string ts_trim(const std::string& s) {
    const std::size_t first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const std::size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

inline std::vector<std::string> ts_split_ws(const std::string& s) {
    std::istringstream ss(s);
    std::vector<std::string> parts;
    std::string tok;
    while (ss >> tok) {
        parts.push_back(tok);
    }
    return parts;
}

inline std::string ts_parse_meta_line(const std::string& line, const std::string& key) {
    const std::string prefix = "% " + key + ":";
    const std::string trimmed = ts_trim(line);
    if (trimmed.rfind(prefix, 0) != 0) {
        throw std::runtime_error("TensorSuite header expected '" + prefix + "', got: " + trimmed);
    }
    return ts_trim(trimmed.substr(prefix.size()));
}

inline bool ts_host_is_little_endian() {
    const std::uint16_t v = 1;
    return *reinterpret_cast<const std::uint8_t*>(&v) == 1;
}

inline std::uint32_t ts_bswap32(std::uint32_t x) {
    return ((x & 0x000000FFu) << 24) |
           ((x & 0x0000FF00u) << 8) |
           ((x & 0x00FF0000u) >> 8) |
           ((x & 0xFF000000u) >> 24);
}

inline std::uint64_t ts_bswap64(std::uint64_t x) {
    return ((x & 0x00000000000000FFull) << 56) |
           ((x & 0x000000000000FF00ull) << 40) |
           ((x & 0x0000000000FF0000ull) << 24) |
           ((x & 0x00000000FF000000ull) << 8) |
           ((x & 0x000000FF00000000ull) >> 8) |
           ((x & 0x0000FF0000000000ull) >> 24) |
           ((x & 0x00FF000000000000ull) >> 40) |
           ((x & 0xFF00000000000000ull) >> 56);
}

inline std::string ts_opt_string(const mini_json::Value& obj, const std::string& key, const std::string& def = "") {
    if (!obj.contains(key)) return def;
    const auto& v = obj.at(key);
    if (v.is_null()) return def;
    return v.as_string();
}

inline std::vector<char> ts_resolve_labels(const std::vector<char>& labels, int order) {
    if (!labels.empty()) {
        if (static_cast<int>(labels.size()) != order) {
            throw std::runtime_error(
                "TensorSuite mode label count (" + std::to_string(labels.size()) +
                ") does not match tensor order (" + std::to_string(order) + ")");
        }
        return labels;
    }
    if (order > 26) {
        throw std::runtime_error("Cannot assign default single-char labels for order > 26");
    }
    std::vector<char> out;
    out.reserve(static_cast<std::size_t>(order));
    for (int i = 0; i < order; ++i) {
        out.push_back(static_cast<char>('a' + i));
    }
    return out;
}

struct TensorSuiteBlockHeader {
    std::string magic;
    std::string version;
    std::string name;
    int order = 0;
    std::vector<std::int64_t> dimensions;
    std::int64_t nnz = 0;
    std::vector<int> block_partitions;
    std::int64_t nnz_block = 0;
    std::vector<std::vector<std::int64_t>> partition_sizes;
};

inline TensorSuiteBlockHeader ts_parse_block_header_stream(
    std::istream& in,
    const std::string& expected_magic)
{
    TensorSuiteBlockHeader hdr;
    std::string line;

    if (!std::getline(in, line)) {
        throw std::runtime_error("Unexpected EOF reading TensorSuite magic line");
    }
    hdr.magic = ts_trim(line);
    if (hdr.magic != expected_magic) {
        throw std::runtime_error("Invalid TensorSuite magic line: expected '" +
                                 expected_magic + "', got '" + hdr.magic + "'");
    }

    if (!std::getline(in, line)) {
        throw std::runtime_error("Unexpected EOF reading TensorSuite version line");
    }
    hdr.version = ts_parse_meta_line(line, "version");

    if (!std::getline(in, line)) {
        throw std::runtime_error("Unexpected EOF reading TensorSuite name line");
    }
    hdr.name = ts_parse_meta_line(line, "name");

    // Line 4: <order> <dim_0> ... <dim_{order-1}> <nnz>
    if (!std::getline(in, line)) {
        throw std::runtime_error("Unexpected EOF reading TensorSuite size line");
    }
    {
        const auto parts = ts_split_ws(line);
        if (parts.size() < 3) {
            throw std::runtime_error("TensorSuite size line must contain order, dimensions, and nnz");
        }
        const std::int64_t order64 = std::stoll(parts[0]);
        if (order64 <= 0 || order64 > 64) {
            throw std::runtime_error("Invalid TensorSuite tensor order: " + parts[0]);
        }
        hdr.order = static_cast<int>(order64);
        if (static_cast<int>(parts.size()) != hdr.order + 2) {
            throw std::runtime_error("TensorSuite size line field count mismatch");
        }
        hdr.dimensions.reserve(static_cast<std::size_t>(hdr.order));
        for (int d = 0; d < hdr.order; ++d) {
            const std::int64_t dim = std::stoll(parts[static_cast<std::size_t>(d + 1)]);
            if (dim <= 0) {
                throw std::runtime_error("TensorSuite dimension must be positive");
            }
            hdr.dimensions.push_back(dim);
        }
        hdr.nnz = std::stoll(parts.back());
        if (hdr.nnz < 0) {
            throw std::runtime_error("TensorSuite nnz must be nonnegative");
        }
    }

    // Line 5: <b_0> ... <b_{order-1}> <nnz_block>
    if (!std::getline(in, line)) {
        throw std::runtime_error("Unexpected EOF reading TensorSuite block grid line");
    }
    {
        const auto parts = ts_split_ws(line);
        if (static_cast<int>(parts.size()) != hdr.order + 1) {
            throw std::runtime_error("TensorSuite block grid line must contain order block counts plus nnz_block");
        }
        hdr.block_partitions.reserve(static_cast<std::size_t>(hdr.order));
        for (int d = 0; d < hdr.order; ++d) {
            const std::int64_t nb = std::stoll(parts[static_cast<std::size_t>(d)]);
            if (nb <= 0) {
                throw std::runtime_error("TensorSuite block partition count must be positive");
            }
            hdr.block_partitions.push_back(static_cast<int>(nb));
        }
        hdr.nnz_block = std::stoll(parts.back());
        if (hdr.nnz_block < 0) {
            throw std::runtime_error("TensorSuite nnz_block must be nonnegative");
        }
    }

    // Next `order` lines: partition sizes for each mode
    hdr.partition_sizes.resize(static_cast<std::size_t>(hdr.order));
    for (int d = 0; d < hdr.order; ++d) {
        if (!std::getline(in, line)) {
            throw std::runtime_error("Unexpected EOF reading TensorSuite block partition sizes for mode " +
                                     std::to_string(d));
        }
        const auto parts = ts_split_ws(line);
        const int expected_count = hdr.block_partitions[static_cast<std::size_t>(d)];
        if (static_cast<int>(parts.size()) != expected_count) {
            throw std::runtime_error(
                "TensorSuite partition size count mismatch on mode " + std::to_string(d) +
                ": expected " + std::to_string(expected_count) +
                ", got " + std::to_string(parts.size()));
        }
        auto& mode_sizes = hdr.partition_sizes[static_cast<std::size_t>(d)];
        mode_sizes.reserve(parts.size());
        std::int64_t sum = 0;
        for (const auto& tok : parts) {
            const std::int64_t sz = std::stoll(tok);
            if (sz <= 0) {
                throw std::runtime_error("TensorSuite partition size must be positive");
            }
            mode_sizes.push_back(sz);
            sum += sz;
        }
        if (sum != hdr.dimensions[static_cast<std::size_t>(d)]) {
            throw std::runtime_error(
                "TensorSuite partition sizes on mode " + std::to_string(d) +
                " sum to " + std::to_string(sum) +
                ", expected " + std::to_string(hdr.dimensions[static_cast<std::size_t>(d)]));
        }
    }

    return hdr;
}

inline void ts_cross_check_header_metadata(
    const TensorSuiteBlockHeader& hdr,
    const TensorSuiteMetadata& meta)
{
    if (meta.sparsity_type != "block") {
        throw std::runtime_error(
            "TensorSuite tensor '" + meta.name + "' has sparsity_type='" + meta.sparsity_type +
            "', but TIDES only supports sparsity_type='block'");
    }
    if (!meta.version.empty() && hdr.version != meta.version) {
        throw std::runtime_error("TensorSuite version mismatch between header and metadata");
    }
    if (!meta.name.empty() && hdr.name != meta.name) {
        throw std::runtime_error("TensorSuite name mismatch between header and metadata");
    }
    if (meta.order != hdr.order) {
        throw std::runtime_error("TensorSuite order mismatch between header and metadata");
    }
    if (meta.dimensions != hdr.dimensions) {
        throw std::runtime_error("TensorSuite dimensions mismatch between header and metadata");
    }
    if (meta.nnz != hdr.nnz) {
        throw std::runtime_error("TensorSuite nnz mismatch between header and metadata");
    }
    if (!meta.has_block_partitions || meta.block_partitions != hdr.block_partitions) {
        throw std::runtime_error("TensorSuite block_partitions mismatch between header and metadata");
    }
    if (!meta.has_nnz_block || meta.nnz_block != hdr.nnz_block) {
        throw std::runtime_error("TensorSuite nnz_block mismatch between header and metadata");
    }
}

inline BlockSparseTensor ts_build_tensor(
    const TensorSuiteBlockHeader& hdr,
    const std::set<std::vector<std::int64_t>>& block_coords,
    const std::vector<char>& labels,
    DataType dtype,
    const TensorSuiteMetadata* meta,
    bool skip_values)
{
    const auto resolved_labels = ts_resolve_labels(labels, hdr.order);

    TensorMeta tmeta(
        hdr.name,
        static_cast<std::int16_t>(hdr.order),
        resolved_labels,
        dtype);

    std::map<char, std::vector<std::int64_t>> block_size_map;
    for (int d = 0; d < hdr.order; ++d) {
        block_size_map[resolved_labels[static_cast<std::size_t>(d)]] =
            hdr.partition_sizes[static_cast<std::size_t>(d)];
    }

    BlockSparseTensor tensor(tmeta, hdr.block_partitions, block_size_map, block_coords);
    tensor.host_data.set_data_type(dtype);

    const std::size_t total_block_elems = block_utils::total_elements(tensor);
    if (static_cast<std::int64_t>(total_block_elems) != hdr.nnz) {
        throw std::runtime_error(
            "TensorSuite nnz mismatch: header says " + std::to_string(hdr.nnz) +
            ", but sum of nonzero block volumes is " + std::to_string(total_block_elems));
    }

    if (!skip_values && total_block_elems > 0) {
        if (dtype != DataType::FLOAT32 && dtype != DataType::FLOAT64) {
            throw std::runtime_error(
                "TensorSuite host value initialization only supports FLOAT32 and FLOAT64");
        }
        tensor.host_data.allocate(total_block_elems);
        const std::string domain = meta ? meta->value_domain : "general";
        if (domain == "binary") {
            utils::Xorshift32 rng(42u);
            if (dtype == DataType::FLOAT64) {
                double* ptr = tensor.host_data.data_as<double>();
                for (std::size_t i = 0; i < total_block_elems; ++i) {
                    ptr[i] = (rng.next() & 1u) ? 1.0 : 0.0;
                }
            } else {
                float* ptr = tensor.host_data.data_as<float>();
                for (std::size_t i = 0; i < total_block_elems; ++i) {
                    ptr[i] = (rng.next() & 1u) ? 1.0f : 0.0f;
                }
            }
        } else {
            if (dtype == DataType::FLOAT64) {
                utils::init_random<double>(tensor, 0.0, 1.0);
            } else {
                utils::init_random<float>(tensor, 0.0f, 1.0f);
            }
        }
    }

    return tensor;
}

} // namespace detail

// Parse a TensorSuite *_metadata.json file.
inline TensorSuiteMetadata parse_tensorsuite_metadata(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open TensorSuite metadata JSON: " + json_path);
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();

    const mini_json::Value root = mini_json::parse(buffer.str());
    if (!root.is_object()) {
        throw std::runtime_error("TensorSuite metadata JSON root must be an object: " + json_path);
    }

    TensorSuiteMetadata meta;
    meta.version = detail::ts_opt_string(root, "version");
    meta.name = detail::ts_opt_string(root, "name");
    meta.group = detail::ts_opt_string(root, "group");
    meta.id = detail::ts_opt_string(root, "id");
    meta.time = detail::ts_opt_string(root, "time");
    meta.source_type = detail::ts_opt_string(root, "source_type");
    meta.source = detail::ts_opt_string(root, "source");
    meta.source_url = detail::ts_opt_string(root, "source_url");

    meta.value_type = detail::ts_opt_string(root, "value_type", "float64");
    meta.value_domain = detail::ts_opt_string(root, "value_domain", "general");
    if (root.contains("values_provided") && !root.at("values_provided").is_null()) {
        meta.values_provided = root.at("values_provided").as_bool();
    }

    meta.index_type = detail::ts_opt_string(root, "index_type");
    meta.endianness = detail::ts_opt_string(root, "endianness");
    if (!root.contains("index_base") || root.at("index_base").is_null()) {
        throw std::runtime_error("TensorSuite metadata JSON missing required 'index_base': " + json_path);
    }
    meta.index_base = static_cast<int>(root.at("index_base").as_int());
    if (meta.index_base != 0 && meta.index_base != 1) {
        throw std::runtime_error("TensorSuite metadata index_base must be 0 or 1");
    }

    meta.sorted = detail::ts_opt_string(root, "sorted");
    meta.duplicates = detail::ts_opt_string(root, "duplicates", "disallowed");
    meta.explicit_zeros = detail::ts_opt_string(root, "explicit_zeros");
    meta.pattern_symmetry = detail::ts_opt_string(root, "pattern_symmetry");
    meta.numerical_symmetry = detail::ts_opt_string(root, "numerical_symmetry");
    meta.sparsity_type = detail::ts_opt_string(root, "sparsity_type");

    if (root.contains("order") && !root.at("order").is_null()) {
        meta.order = static_cast<int>(root.at("order").as_int());
    }
    if (root.contains("dimensions") && !root.at("dimensions").is_null()) {
        for (const auto& elem : root.at("dimensions").as_array()) {
            meta.dimensions.push_back(elem.as_int());
        }
    }
    if (root.contains("nnz") && !root.at("nnz").is_null()) {
        meta.nnz = root.at("nnz").as_int();
    }

    if (root.contains("block_partitions") && !root.at("block_partitions").is_null()) {
        meta.has_block_partitions = true;
        for (const auto& elem : root.at("block_partitions").as_array()) {
            meta.block_partitions.push_back(static_cast<int>(elem.as_int()));
        }
    }
    if (root.contains("nnz_block") && !root.at("nnz_block").is_null()) {
        meta.has_nnz_block = true;
        meta.nnz_block = root.at("nnz_block").as_int();
    }

    if (root.contains("files") && root.at("files").is_object()) {
        const auto& files = root.at("files");
        meta.file_text = detail::ts_opt_string(files, "text");
        meta.file_binary = detail::ts_opt_string(files, "binary");
        meta.file_readme = detail::ts_opt_string(files, "readme");
    }

    return meta;
}

// Locate sibling or bundle *_metadata.json for a given path (bundle dir, .tns, .tnsb, or .json).
inline std::string find_tensorsuite_metadata_path(const std::string& path_str) {
    namespace fs = std::filesystem;
    const fs::path p(path_str);

    if (fs::is_regular_file(p) && p.extension() == ".json") {
        return p.string();
    }

    if (fs::is_directory(p)) {
        const fs::path by_dir = p / (p.filename().string() + "_metadata.json");
        if (fs::is_regular_file(by_dir)) {
            return by_dir.string();
        }
        std::string found;
        for (const auto& entry : fs::directory_iterator(p)) {
            if (!entry.is_regular_file()) continue;
            const std::string fname = entry.path().filename().string();
            const std::string suffix = "_metadata.json";
            if (fname.size() >= suffix.size() &&
                fname.compare(fname.size() - suffix.size(), suffix.size(), suffix) == 0) {
                if (!found.empty()) {
                    throw std::runtime_error("Multiple *_metadata.json files found in bundle: " + path_str);
                }
                found = entry.path().string();
            }
        }
        return found;
    }

    const fs::path parent = p.parent_path();
    const fs::path by_stem = parent / (p.stem().string() + "_metadata.json");
    if (fs::is_regular_file(by_stem)) {
        return by_stem.string();
    }
    if (!parent.empty()) {
        const fs::path by_parent = parent / (parent.filename().string() + "_metadata.json");
        if (fs::is_regular_file(by_parent)) {
            return by_parent.string();
        }
    }
    return "";
}

// Parse a TensorSuite block-sparse text (.tns) file.
// Metadata is required to determine sparsity_type and index_base (0 or 1).
inline BlockSparseTensor parse_tensorsuite_block_tns(
    const std::string& tns_path,
    DataType dtype,
    const std::vector<char>& labels,
    const TensorSuiteMetadata* meta,
    bool skip_values = false)
{
    if (!meta) {
        throw std::runtime_error(
            "TensorSuite .tns parsing requires metadata (*_metadata.json) to determine sparsity_type and index_base: " +
            tns_path);
    }

    std::ifstream file(tns_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open TensorSuite .tns file: " + tns_path);
    }

    const auto hdr = detail::ts_parse_block_header_stream(file, "%%TensorSuite-TNS");
    detail::ts_cross_check_header_metadata(hdr, *meta);

    const int index_base = meta->index_base;
    const bool disallow_dups = meta->duplicates != "allowed";

    std::set<std::vector<std::int64_t>> block_coords;
    std::int64_t seen = 0;
    std::string line;
    while (std::getline(file, line)) {
        const std::string trimmed = detail::ts_trim(line);
        if (trimmed.empty()) continue;

        const auto parts = detail::ts_split_ws(trimmed);
        if (static_cast<int>(parts.size()) != hdr.order) {
            throw std::runtime_error(
                "TensorSuite .tns block coordinate rank mismatch: expected " +
                std::to_string(hdr.order) + ", got " + std::to_string(parts.size()));
        }

        std::vector<std::int64_t> coord(static_cast<std::size_t>(hdr.order));
        for (int d = 0; d < hdr.order; ++d) {
            const std::int64_t raw = std::stoll(parts[static_cast<std::size_t>(d)]);
            const std::int64_t zero_based = raw - static_cast<std::int64_t>(index_base);
            if (zero_based < 0 || zero_based >= hdr.block_partitions[static_cast<std::size_t>(d)]) {
                throw std::runtime_error(
                    "TensorSuite .tns block coordinate out of range on mode " + std::to_string(d) +
                    ": raw=" + std::to_string(raw) + ", index_base=" + std::to_string(index_base));
            }
            coord[static_cast<std::size_t>(d)] = zero_based;
        }

        const auto inserted = block_coords.insert(std::move(coord));
        if (!inserted.second && disallow_dups) {
            throw std::runtime_error("Duplicate block coordinate in TensorSuite .tns file: " + tns_path);
        }
        ++seen;
    }

    if (seen != hdr.nnz_block) {
        throw std::runtime_error(
            "TensorSuite .tns nnz_block mismatch: header=" + std::to_string(hdr.nnz_block) +
            ", actual=" + std::to_string(seen));
    }

    return detail::ts_build_tensor(hdr, block_coords, labels, dtype, meta, skip_values);
}

// Parse a TensorSuite block-sparse binary (.tnsb) file.
// Metadata is required to determine sparsity_type, index_base, index_type, and endianness.
inline BlockSparseTensor parse_tensorsuite_block_tnsb(
    const std::string& tnsb_path,
    DataType dtype,
    const std::vector<char>& labels,
    const TensorSuiteMetadata* meta,
    bool skip_values = false)
{
    if (!meta) {
        throw std::runtime_error(
            "TensorSuite .tnsb parsing requires metadata (*_metadata.json) to determine index_base, index_type, and endianness: " +
            tnsb_path);
    }

    std::ifstream file(tnsb_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open TensorSuite .tnsb file: " + tnsb_path);
    }

    const auto hdr = detail::ts_parse_block_header_stream(file, "%%TensorSuite-TNSB");
    detail::ts_cross_check_header_metadata(hdr, *meta);

    const std::string& index_type = meta->index_type;
    const std::string& endianness = meta->endianness;
    const int index_base = meta->index_base;
    const bool disallow_dups = meta->duplicates != "allowed";

    if (index_type != "uint32" && index_type != "uint64") {
        throw std::runtime_error("Unsupported or missing TensorSuite .tnsb index_type: '" + index_type + "'");
    }
    if (endianness != "little" && endianness != "big") {
        throw std::runtime_error("Unsupported or missing TensorSuite .tnsb endianness: '" + endianness + "'");
    }

    const std::size_t elem_size = (index_type == "uint32") ? sizeof(std::uint32_t) : sizeof(std::uint64_t);
    const std::streamoff payload_offset = file.tellg();
    file.seekg(0, std::ios::end);
    const std::streamoff file_size = file.tellg();
    file.seekg(payload_offset, std::ios::beg);

    const std::int64_t payload_bytes = static_cast<std::int64_t>(file_size - payload_offset);
    const std::int64_t expected_bytes =
        hdr.nnz_block * static_cast<std::int64_t>(hdr.order) * static_cast<std::int64_t>(elem_size);
    if (payload_bytes != expected_bytes) {
        throw std::runtime_error(
            "TensorSuite .tnsb payload size mismatch: expected " + std::to_string(expected_bytes) +
            " bytes, got " + std::to_string(payload_bytes));
    }

    const bool need_swap = (endianness == "big") == detail::ts_host_is_little_endian();

    std::set<std::vector<std::int64_t>> block_coords;
    std::vector<std::int64_t> coord(static_cast<std::size_t>(hdr.order));

    for (std::int64_t b = 0; b < hdr.nnz_block; ++b) {
        for (int d = 0; d < hdr.order; ++d) {
            std::uint64_t raw = 0;
            if (index_type == "uint32") {
                std::uint32_t v = 0;
                file.read(reinterpret_cast<char*>(&v), sizeof(v));
                if (!file) throw std::runtime_error("Unexpected EOF in TensorSuite .tnsb payload");
                if (need_swap) v = detail::ts_bswap32(v);
                raw = static_cast<std::uint64_t>(v);
            } else {
                std::uint64_t v = 0;
                file.read(reinterpret_cast<char*>(&v), sizeof(v));
                if (!file) throw std::runtime_error("Unexpected EOF in TensorSuite .tnsb payload");
                if (need_swap) v = detail::ts_bswap64(v);
                raw = v;
            }

            if (raw < static_cast<std::uint64_t>(index_base)) {
                throw std::runtime_error("TensorSuite .tnsb coordinate smaller than index_base");
            }
            const std::uint64_t zero_based = raw - static_cast<std::uint64_t>(index_base);
            if (zero_based >= static_cast<std::uint64_t>(hdr.block_partitions[static_cast<std::size_t>(d)])) {
                throw std::runtime_error(
                    "TensorSuite .tnsb block coordinate out of range on mode " + std::to_string(d));
            }
            coord[static_cast<std::size_t>(d)] = static_cast<std::int64_t>(zero_based);
        }

        const auto inserted = block_coords.insert(coord);
        if (!inserted.second && disallow_dups) {
            throw std::runtime_error("Duplicate block coordinate in TensorSuite .tnsb file: " + tnsb_path);
        }
    }

    return detail::ts_build_tensor(hdr, block_coords, labels, dtype, meta, skip_values);
}

// Parse a TensorSuite block-sparse tensor from a bundle directory, *_metadata.json, .tns, or .tnsb path.
inline BlockSparseTensor parse_tensorsuite_tensor(
    const std::string& path_str,
    DataType dtype,
    const std::vector<char>& labels = {},
    bool skip_values = false)
{
    namespace fs = std::filesystem;
    const fs::path input_path(path_str);
    if (!fs::exists(input_path)) {
        throw std::runtime_error("TensorSuite path does not exist: " + path_str);
    }

    const std::string meta_path = find_tensorsuite_metadata_path(path_str);
    if (meta_path.empty()) {
        throw std::runtime_error(
            "Missing TensorSuite *_metadata.json for '" + path_str +
            "'; metadata is required to determine sparsity_type, index_base, index_type, and endianness");
    }
    const TensorSuiteMetadata meta_storage = parse_tensorsuite_metadata(meta_path);
    const TensorSuiteMetadata* meta_ptr = &meta_storage;

    fs::path data_path;
    if (fs::is_directory(input_path) || input_path.extension() == ".json") {
        const fs::path bundle_dir = fs::is_directory(input_path) ? input_path : input_path.parent_path();
        if (meta_ptr) {
            if (!meta_ptr->file_binary.empty() && fs::is_regular_file(bundle_dir / meta_ptr->file_binary)) {
                data_path = bundle_dir / meta_ptr->file_binary;
            } else if (!meta_ptr->file_text.empty() && fs::is_regular_file(bundle_dir / meta_ptr->file_text)) {
                data_path = bundle_dir / meta_ptr->file_text;
            }
        }
        if (data_path.empty()) {
            const std::string base = bundle_dir.filename().string();
            if (fs::is_regular_file(bundle_dir / (base + ".tnsb"))) {
                data_path = bundle_dir / (base + ".tnsb");
            } else if (fs::is_regular_file(bundle_dir / (base + ".tns"))) {
                data_path = bundle_dir / (base + ".tns");
            } else {
                throw std::runtime_error("No .tns or .tnsb data file found in TensorSuite bundle: " + path_str);
            }
        }
    } else {
        data_path = input_path;
    }

    // Detect text vs binary from magic line (preferred over extension alone).
    std::ifstream probe(data_path, std::ios::binary);
    if (!probe.is_open()) {
        throw std::runtime_error("Could not open TensorSuite data file: " + data_path.string());
    }
    std::string first_line;
    std::getline(probe, first_line);
    const std::string magic = detail::ts_trim(first_line);

    if (magic == "%%TensorSuite-TNSB") {
        return parse_tensorsuite_block_tnsb(data_path.string(), dtype, labels, meta_ptr, skip_values);
    }
    if (magic == "%%TensorSuite-TNS") {
        return parse_tensorsuite_block_tns(data_path.string(), dtype, labels, meta_ptr, skip_values);
    }
    throw std::runtime_error("Unrecognized TensorSuite magic header in '" + data_path.string() + "': " + magic);
}

// Relabel a BlockSparseTensor's mode indices (e.g., to match an einsum expression).
inline void relabel_block_sparse_tensor(
    BlockSparseTensor& tensor,
    const std::vector<char>& new_labels)
{
    if (static_cast<int>(new_labels.size()) != tensor.meta.ndim) {
        throw std::runtime_error(
            "relabel_block_sparse_tensor: label count (" + std::to_string(new_labels.size()) +
            ") does not match tensor rank (" + std::to_string(tensor.meta.ndim) + ")");
    }
    std::map<char, std::vector<std::int64_t>> new_block_sizes;
    for (std::size_t d = 0; d < new_labels.size(); ++d) {
        const char old_lbl = tensor.meta.indices[d];
        const char new_lbl = new_labels[d];
        auto it = tensor.block_sizes.find(old_lbl);
        if (it == tensor.block_sizes.end()) {
            throw std::runtime_error("relabel_block_sparse_tensor: missing block_sizes entry");
        }
        new_block_sizes[new_lbl] = std::move(it->second);
    }
    tensor.meta.indices = new_labels;
    tensor.block_sizes = std::move(new_block_sizes);
}

} // namespace bsptc
