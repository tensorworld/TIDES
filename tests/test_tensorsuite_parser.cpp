#include "bs_parser.hpp"
#include "mini_json.hpp"
#include "tensorsuite_parser.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace bsptc;
namespace fs = std::filesystem;

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__ << "]: " #cond << "\n";    \
            ++g_failed;                                                                     \
            return;                                                                         \
        }                                                                                   \
    } while (0)

#define CHECK_THROWS(expr)                                                                  \
    do {                                                                                    \
        bool threw = false;                                                                 \
        try {                                                                               \
            (void)(expr);                                                                   \
        } catch (const std::exception&) {                                                   \
            threw = true;                                                                   \
        }                                                                                   \
        if (!threw) {                                                                       \
            std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__ << "]: expected throw: "   \
                      << #expr << "\n";                                                     \
            ++g_failed;                                                                     \
            return;                                                                         \
        }                                                                                   \
    } while (0)

static void run_test(const char* name, void (*fn)()) {
    const int before = g_failed;
    try {
        fn();
    } catch (const std::exception& e) {
        std::cerr << "FAIL [" << name << "]: unexpected exception: " << e.what() << "\n";
        ++g_failed;
    }
    if (g_failed == before) {
        std::cout << "PASS: " << name << "\n";
        ++g_passed;
    }
}

static void write_text_file(const fs::path& p, const std::string& content) {
    fs::create_directories(p.parent_path());
    std::ofstream out(p);
    if (!out.is_open()) throw std::runtime_error("Cannot write: " + p.string());
    out << content;
}

template <typename IntT>
static void write_tnsb_file(
    const fs::path& p,
    const std::string& name,
    const std::vector<std::int64_t>& dims,
    std::int64_t nnz,
    const std::vector<int>& block_partitions,
    const std::vector<std::vector<std::int64_t>>& partition_sizes,
    const std::vector<std::vector<std::uint64_t>>& coords,
    bool big_endian)
{
    fs::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("Cannot write: " + p.string());

    out << "%%TensorSuite-TNSB\n";
    out << "% version: 0.1\n";
    out << "% name: " << name << "\n";
    out << dims.size();
    for (auto d : dims) out << " " << d;
    out << " " << nnz << "\n";
    for (std::size_t i = 0; i < block_partitions.size(); ++i) {
        if (i > 0) out << " ";
        out << block_partitions[i];
    }
    out << " " << coords.size() << "\n";
    for (const auto& ps : partition_sizes) {
        for (std::size_t i = 0; i < ps.size(); ++i) {
            if (i > 0) out << " ";
            out << ps[i];
        }
        out << "\n";
    }

    const bool host_le = detail::ts_host_is_little_endian();
    const bool need_swap = (big_endian == host_le);
    for (const auto& c : coords) {
        for (std::uint64_t raw : c) {
            IntT val = static_cast<IntT>(raw);
            if (need_swap) {
                if constexpr (sizeof(IntT) == 4) {
                    val = detail::ts_bswap32(val);
                } else {
                    val = detail::ts_bswap64(val);
                }
            }
            out.write(reinterpret_cast<const char*>(&val), sizeof(val));
        }
    }
}

static void assert_tensors_equal(const BlockSparseTensor& a, const BlockSparseTensor& b, bool check_values) {
    if (a.meta.ndim != b.meta.ndim) throw std::runtime_error("ndim mismatch");
    if (a.meta.indices != b.meta.indices) throw std::runtime_error("indices mismatch");
    if (a.num_blocks != b.num_blocks) throw std::runtime_error("num_blocks mismatch");
    if (a.block_sizes != b.block_sizes) throw std::runtime_error("block_sizes mismatch");
    if (a.block_coords != b.block_coords) throw std::runtime_error("block_coords mismatch");
    if (check_values) {
        const std::size_t n = block_utils::total_elements(a);
        if (n != block_utils::total_elements(b)) throw std::runtime_error("total_elements mismatch");
        if (a.meta.dtype == DataType::FLOAT64) {
            if (std::memcmp(a.data(), b.data(), n * sizeof(double)) != 0) {
                throw std::runtime_error("float64 values mismatch");
            }
        } else {
            if (std::memcmp(a.data(), b.data(), n * sizeof(float)) != 0) {
                throw std::runtime_error("float32 values mismatch");
            }
        }
    }
}

static void test_mini_json() {
    const std::string doc = R"({
        "version": "0.1",
        "name": "test\"escaped\\slash\/line\n\u0041",
        "values_provided": false,
        "flag_true": true,
        "nullable": null,
        "order": 3,
        "negative": -42,
        "pi": 3.14159e0,
        "dimensions": [1600, 20, 1600],
        "dense_modes": [],
        "files": {
            "text": "a.tns",
            "binary": null
        }
    })";
    const auto v = mini_json::parse(doc);
    CHECK(v.is_object());
    CHECK(v.at("version").as_string() == "0.1");
    CHECK(v.at("name").as_string() == "test\"escaped\\slash/line\nA");
    CHECK(v.at("values_provided").as_bool() == false);
    CHECK(v.at("flag_true").as_bool() == true);
    CHECK(v.at("nullable").is_null());
    CHECK(v.at("order").as_int() == 3);
    CHECK(v.at("negative").as_int() == -42);
    CHECK(std::abs(v.at("pi").as_double() - 3.14159) < 1e-9);
    CHECK(v.at("dimensions").as_array().size() == 3);
    CHECK(v.at("dimensions").as_array()[1].as_int() == 20);
    CHECK(v.at("dense_modes").as_array().empty());
    CHECK(v.at("files").at("text").as_string() == "a.tns");
    CHECK(v.at("files").at("binary").is_null());

    CHECK_THROWS(mini_json::parse("{bad}"));
    CHECK_THROWS(mini_json::parse("{\"a\": 1} trailing"));
    CHECK_THROWS(mini_json::parse("[1, 2,]"));
    CHECK_THROWS(mini_json::parse("\"unterminated"));
}

static void test_tns_0based_and_1based_and_bs_equiv() {
    const fs::path tmp = fs::temp_directory_path() / "tides_ts_test_1";
    fs::remove_all(tmp);

    // Build a 3D block-sparse tensor in .bs and in TensorSuite (0-based & 1-based).
    // Grid: 2 x 3 x 2; partition sizes: [2, 3], [4, 1, 5], [3, 2]
    // Dims: [5, 10, 5]
    // Blocks (0-based): (0, 0, 1) -> vol 2*4*2 = 16; (1, 2, 0) -> vol 3*5*3 = 45; nnz = 61
    const fs::path bs_path = tmp / "ref.bs";
    write_text_file(bs_path,
        "# Tensor ref (ijk)\n"
        "{2, 3, 2}\n"
        "2\n"
        "{2, 3, 4, 1, 5, 3, 2}\n"
        "0, 0, 1\n"
        "1, 2, 0\n");

    const fs::path bundle0 = tmp / "bundle0";
    write_text_file(bundle0 / "bundle0_metadata.json", R"({
        "version": "0.1",
        "name": "bundle0",
        "value_type": "float64",
        "value_domain": "general",
        "values_provided": false,
        "index_type": "uint64",
        "endianness": "none",
        "index_base": 0,
        "duplicates": "disallowed",
        "sparsity_type": "block",
        "order": 3,
        "dimensions": [5, 10, 5],
        "nnz": 61,
        "block_partitions": [2, 3, 2],
        "nnz_block": 2,
        "files": { "text": "bundle0.tns", "readme": "README.md" }
    })");
    write_text_file(bundle0 / "bundle0.tns",
        "%%TensorSuite-TNS\n"
        "% version: 0.1\n"
        "% name: bundle0\n"
        "3 5 10 5 61\n"
        "2 3 2 2\n"
        "2 3\n"
        "4 1 5\n"
        "3 2\n"
        "0 0 1\n"
        "1 2 0\n");

    const fs::path bundle1 = tmp / "bundle1";
    write_text_file(bundle1 / "bundle1_metadata.json", R"({
        "version": "0.1",
        "name": "bundle1",
        "value_type": "float64",
        "value_domain": "nonnegative",
        "values_provided": false,
        "index_type": "uint64",
        "endianness": "none",
        "index_base": 1,
        "duplicates": "disallowed",
        "sparsity_type": "block",
        "order": 3,
        "dimensions": [5, 10, 5],
        "nnz": 61,
        "block_partitions": [2, 3, 2],
        "nnz_block": 2,
        "files": { "text": "bundle1.tns", "readme": "README.md" }
    })");
    write_text_file(bundle1 / "bundle1.tns",
        "%%TensorSuite-TNS\n"
        "% version: 0.1\n"
        "% name: bundle1\n"
        "3 5 10 5 61\n"
        "2 3 2 2\n"
        "2 3\n"
        "4 1 5\n"
        "3 2\n"
        "1 1 2\n"
        "2 3 1\n");

    const std::vector<char> labels = {'i', 'j', 'k'};
    const auto t_bs = parse_tensor_file(bs_path.string(), DataType::FLOAT64, labels);
    const auto t_dir0 = parse_tensor_file(bundle0.string(), DataType::FLOAT64, labels);
    const auto t_tns0 = parse_tensor_file((bundle0 / "bundle0.tns").string(), DataType::FLOAT64, labels);
    const auto t_dir1 = parse_tensor_file(bundle1.string(), DataType::FLOAT64, labels);
    const auto t_json1 = parse_tensor_file((bundle1 / "bundle1_metadata.json").string(), DataType::FLOAT64, labels);

    assert_tensors_equal(t_bs, t_dir0, true);
    assert_tensors_equal(t_bs, t_tns0, true);
    assert_tensors_equal(t_bs, t_dir1, true);
    assert_tensors_equal(t_bs, t_json1, true);

    fs::remove_all(tmp);
}

static void test_tnsb_binary_variants() {
    const fs::path tmp = fs::temp_directory_path() / "tides_ts_test_bin";
    fs::remove_all(tmp);

    const std::vector<std::int64_t> dims = {5, 10, 5};
    const std::vector<int> grid = {2, 3, 2};
    const std::vector<std::vector<std::int64_t>> parts = {{2, 3}, {4, 1, 5}, {3, 2}};
    const std::vector<std::vector<std::uint64_t>> coords0 = {{0, 0, 1}, {1, 2, 0}};
    const std::vector<std::vector<std::uint64_t>> coords1 = {{1, 1, 2}, {2, 3, 1}};
    const std::vector<char> labels = {'x', 'y', 'z'};

    // Case 1: uint64 little-endian 0-based
    const fs::path b_u64_le = tmp / "u64_le";
    write_text_file(b_u64_le / "u64_le_metadata.json", R"({
        "version": "0.1", "name": "u64_le", "sparsity_type": "block",
        "index_type": "uint64", "endianness": "little", "index_base": 0,
        "order": 3, "dimensions": [5, 10, 5], "nnz": 61,
        "block_partitions": [2, 3, 2], "nnz_block": 2,
        "files": { "binary": "u64_le.tnsb" }
    })");
    write_tnsb_file<std::uint64_t>(b_u64_le / "u64_le.tnsb", "u64_le", dims, 61, grid, parts, coords0, false);

    // Case 2: uint32 big-endian 1-based
    const fs::path b_u32_be = tmp / "u32_be";
    write_text_file(b_u32_be / "u32_be_metadata.json", R"({
        "version": "0.1", "name": "u32_be", "sparsity_type": "block",
        "index_type": "uint32", "endianness": "big", "index_base": 1,
        "order": 3, "dimensions": [5, 10, 5], "nnz": 61,
        "block_partitions": [2, 3, 2], "nnz_block": 2,
        "files": { "binary": "u32_be.tnsb" }
    })");
    write_tnsb_file<std::uint32_t>(b_u32_be / "u32_be.tnsb", "u32_be", dims, 61, grid, parts, coords1, true);

    const auto t1 = parse_tensor_file(b_u64_le.string(), DataType::FLOAT32, labels);
    const auto t2 = parse_tensor_file(b_u32_be.string(), DataType::FLOAT32, labels);
    assert_tensors_equal(t1, t2, true);

    fs::remove_all(tmp);
}

static void test_validation_errors() {
    const fs::path tmp = fs::temp_directory_path() / "tides_ts_test_err";
    fs::remove_all(tmp);

    // Element-wise metadata should be rejected
    const fs::path elem_dir = tmp / "elem";
    write_text_file(elem_dir / "elem_metadata.json", R"({
        "version": "0.1", "name": "elem", "sparsity_type": "element",
        "index_base": 1, "order": 2, "dimensions": [4, 4], "nnz": 1,
        "block_partitions": null, "nnz_block": null,
        "files": { "text": "elem.tns" }
    })");
    write_text_file(elem_dir / "elem.tns",
        "%%TensorSuite-TNS\n"
        "% version: 0.1\n"
        "% name: elem\n"
        "2 4 4 1\n"
        "2 2 1\n"
        "2 2\n"
        "2 2\n"
        "1 1\n");
    CHECK_THROWS(parse_tensor_file(elem_dir.string(), DataType::FLOAT32, {'a', 'b'}));

    // Out-of-range coordinate for 1-based indexing (0 is invalid when index_base=1)
    const fs::path oor_dir = tmp / "oor";
    write_text_file(oor_dir / "oor_metadata.json", R"({
        "version": "0.1", "name": "oor", "sparsity_type": "block",
        "index_base": 1, "order": 2, "dimensions": [4, 4], "nnz": 4,
        "block_partitions": [2, 2], "nnz_block": 1,
        "files": { "text": "oor.tns" }
    })");
    write_text_file(oor_dir / "oor.tns",
        "%%TensorSuite-TNS\n"
        "% version: 0.1\n"
        "% name: oor\n"
        "2 4 4 4\n"
        "2 2 1\n"
        "2 2\n"
        "2 2\n"
        "0 1\n");
    CHECK_THROWS(parse_tensor_file(oor_dir.string(), DataType::FLOAT32, {'a', 'b'}));

    // Wrong label count
    const fs::path bad_lbl = tmp / "bad_lbl.tns";
    write_text_file(bad_lbl,
        "%%TensorSuite-TNS\n"
        "% version: 0.1\n"
        "% name: bad_lbl\n"
        "2 4 4 4\n"
        "2 2 1\n"
        "2 2\n"
        "2 2\n"
        "0 1\n");
    CHECK_THROWS(parse_tensor_file(bad_lbl.string(), DataType::FLOAT32, {'a', 'b', 'c'}));

    fs::remove_all(tmp);
}

int main(int argc, char** argv) {
    run_test("mini_json", test_mini_json);
    run_test("tns_0based_and_1based_and_bs_equiv", test_tns_0based_and_1based_and_bs_equiv);
    run_test("tnsb_binary_variants", test_tnsb_binary_variants);
    run_test("validation_errors", test_validation_errors);

    // Optional CLI mode: compare an external TensorSuite bundle against a reference .bs file:
    //   test_tensorsuite_parser --compare <ts_path> <bs_path> <labels>
    if (argc == 5 && std::string(argv[1]) == "--compare") {
        const std::string ts_path = argv[2];
        const std::string bs_path = argv[3];
        const std::string lbl_str = argv[4];
        const std::vector<char> labels(lbl_str.begin(), lbl_str.end());
        try {
            const auto t_ts = parse_tensor_file(ts_path, DataType::FLOAT64, labels);
            const auto t_bs = parse_bs_tensor_file(bs_path, DataType::FLOAT64);
            assert_tensors_equal(t_ts, t_bs, true);
            std::cout << "PASS: compare " << ts_path << " == " << bs_path
                      << " (" << t_ts.block_coords.size() << " blocks, "
                      << block_utils::total_elements(t_ts) << " elems)\n";
            ++g_passed;
        } catch (const std::exception& e) {
            std::cerr << "FAIL: compare " << ts_path << " vs " << bs_path << ": " << e.what() << "\n";
            ++g_failed;
        }
    }

    std::cout << "\nSummary: " << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed == 0 ? 0 : 1;
}
