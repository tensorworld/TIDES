#include "bs_parser.hpp"
#include "einsum.hpp"
#include "general_sparse/gpu_sparse_input.hpp"
#include "general_sparse/general_sparse_executor.cuh"
#include "general_sparse/general_sparse_cli.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace bsptc;

template <typename T>
int run_general_sparse_test(const GeneralSparseOptions& opts) {
    // Parse einsum expression first so mode labels can be supplied to formats
    // (like TensorSuite) that do not store contraction labels in the file.
    std::cout << "Parsing einsum: " << opts.einsum << "\n";
    auto expr = parse_einsum(opts.einsum);
    if (expr.inputs.size() != 2) {
        throw std::runtime_error("Expected binary einsum with 2 inputs, got " +
                                 std::to_string(expr.inputs.size()));
    }

    std::vector<char> labelsA(expr.inputs[0].labels.begin(), expr.inputs[0].labels.end());
    std::vector<char> labelsB(expr.inputs[1].labels.begin(), expr.inputs[1].labels.end());

    // Parse input tensors (.bs or TensorSuite block-sparse bundle/.tns/.tnsb)
    std::cout << "Loading tensors...\n";
    auto tensors = parse_tensor_files({opts.tensorA, opts.tensorB}, opts.dtype, {labelsA, labelsB});
    BlockSparseTensor& A = tensors[0];
    BlockSparseTensor& B = tensors[1];

    std::cout << "  Tensor A: " << A.meta.name << " ("
              << std::string(A.meta.indices.begin(), A.meta.indices.end()) << ") "
              << A.block_coords.size() << " blocks\n";
    std::cout << "  Tensor B: " << B.meta.name << " ("
              << std::string(B.meta.indices.begin(), B.meta.indices.end()) << ") "
              << B.block_coords.size() << " blocks\n";

    // Validate shapes
    std::vector<std::vector<int64_t>> shapes = {
        block_utils::compute_full_shape(A),
        block_utils::compute_full_shape(B)
    };
    validate_einsum(expr, shapes);

    // Create index sets
    EinsumIndexSets index_sets;
    create_einsum_index_sets(expr, index_sets);

    if (opts.verbose) {
        std::cout << "Index classification:\n";
        std::cout << "  Free A: ";
        for (char c : index_sets.free_indices_A) std::cout << c;
        std::cout << "\n  Free B: ";
        for (char c : index_sets.free_indices_B) std::cout << c;
        std::cout << "\n  Contracted: ";
        for (char c : index_sets.contracted_indices) std::cout << c;
        std::cout << "\n  Batch: ";
        for (char c : index_sets.batch_indices) std::cout << c;
        std::cout << "\n";
    }

    // Convert to GPU-friendly format
    std::cout << "Converting to GPU format...\n";
    auto A_gpu = to_gpu_input<T>(A);
    auto B_gpu = to_gpu_input<T>(B);

    // Execute general sparse contraction
    std::cout << "Executing general sparse contraction...\n";
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    DispatchConfig dc;
    auto A_dev = copy_sparse_to_device(A_gpu);
    auto B_dev = copy_sparse_to_device(B_gpu);
    cudaEventRecord(start);
    auto result = execute_general_sparse_pipeline(A_gpu, B_gpu, A_dev, B_dev, index_sets, expr.output_labels, dc);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    std::cout << "  Result: " << result.num_output_blocks << " output blocks, "
              << result.total_output_elements << " total elements\n";
    std::cout << "  Kernel time: " << kernel_ms << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (result.total_output_elements == 0) {
        std::cout << "Warning: No output produced (no matching block pairs)\n";
    }

    return 0;
}

int main(int argc, char** argv) {
    try {
        auto opts = parse_general_sparse_cli(argc, argv);

        // Print device info
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        std::cout << "Using GPU: " << prop.name
                  << " (SM " << prop.major << "." << prop.minor << ")\n";

        // Run test based on dtype
        switch (opts.dtype) {
            case DataType::FLOAT32:
                return run_general_sparse_test<float>(opts);
            case DataType::FLOAT64:
                return run_general_sparse_test<double>(opts);
            default:
                std::cerr << "Unsupported dtype (only f32/f64 supported)\n";
                return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
