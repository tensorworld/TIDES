#pragma once

#include "../types.hpp"

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

// Minimal CLI for the general sparse test program.
namespace bsptc {

struct GeneralSparseOptions {
    std::string einsum = "ij,jk->ik";
    std::string tensorA = "tensor_A.bs";
    std::string tensorB = "tensor_B.bs";
    DataType dtype = DataType::FLOAT32;
    bool verbose = false;
};

inline void print_general_sparse_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <einsum>\n"
              << "Options:\n"
              << "  --tensorA <path>       Path to tensor A (.bs, .tns, .tnsb, or TensorSuite bundle dir)\n"
              << "  --tensorB <path>       Path to tensor B (.bs, .tns, .tnsb, or TensorSuite bundle dir)\n"
              << "  --dtype <f16|f32|f64>  Element type (default: f32)\n"
              << "  --einsum <expr>        Einsum expression (positional also supported)\n"
              << "  --verbose              Print detailed output\n"
              << "  -h, --help             Show this message\n";
}

inline DataType parse_general_sparse_dtype(std::string value) {
    for (auto& c : value) c = static_cast<char>(std::tolower(c));
    if (value == "f16" || value == "float16") return DataType::FLOAT16;
    if (value == "f32" || value == "float32") return DataType::FLOAT32;
    if (value == "f64" || value == "float64") return DataType::FLOAT64;
    throw std::runtime_error("Unknown dtype: " + value);
}

inline GeneralSparseOptions parse_general_sparse_cli(int argc, char** argv) {
    GeneralSparseOptions opts;
    bool einsum_set = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto require_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_general_sparse_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--tensorA" || arg == "-A") {
            opts.tensorA = require_value("--tensorA");
        } else if (arg == "--tensorB" || arg == "-B") {
            opts.tensorB = require_value("--tensorB");
        } else if (arg == "--dtype") {
            opts.dtype = parse_general_sparse_dtype(require_value("--dtype"));
        } else if (arg == "--einsum") {
            opts.einsum = require_value("--einsum");
            einsum_set = true;
        } else if (arg == "--verbose" || arg == "-v") {
            opts.verbose = true;
        } else if (!arg.empty() && arg[0] != '-') {
            if (einsum_set) {
                throw std::runtime_error("Unexpected positional argument: " + arg);
            }
            opts.einsum = arg;
            einsum_set = true;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return opts;
}

} // namespace bsptc
