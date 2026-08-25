cmake_minimum_required(VERSION 3.23)

# Default to RelWithDebInfo (-O2 -g) if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# Target architecture selection.
# Must be resolved BEFORE project(): enabling the CUDA language caches
# CMAKE_CUDA_ARCHITECTURES from the nvcc default (e.g. 75 on CUDA 13),
# which would silently override the BSPTC_ARCH mapping below.
set(BSPTC_ARCH "sm80" CACHE STRING "Target GPU architecture (sm80, sm90, or sm120).")
set_property(CACHE BSPTC_ARCH PROPERTY STRINGS "sm80" "sm90" "sm120")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  if(BSPTC_ARCH STREQUAL "sm120")
    set(CMAKE_CUDA_ARCHITECTURES 120)
  elseif(BSPTC_ARCH STREQUAL "sm90")
    set(CMAKE_CUDA_ARCHITECTURES 90)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 80)
  endif()
endif()

project(bsptc LANGUAGES C CXX CUDA)
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES} (BSPTC_ARCH=${BSPTC_ARCH})")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

# Find CUTLASS and cuTENSOR
set(_local_cutlass_root "${CMAKE_CURRENT_SOURCE_DIR}/cutlass")
set(_sibling_cutlass_root "${CMAKE_CURRENT_SOURCE_DIR}/../cutlass")
set(_default_cutlass_root "${_local_cutlass_root}")
if(NOT EXISTS "${_default_cutlass_root}" AND EXISTS "${_sibling_cutlass_root}")
  set(_default_cutlass_root "${_sibling_cutlass_root}")
endif()
set(CUTLASS_ROOT "${_default_cutlass_root}" CACHE PATH "Path to CUTLASS.")

set(_local_cutensor_root "${CMAKE_CURRENT_SOURCE_DIR}/libcutensor")
set(_sibling_cutensor_root "${CMAKE_CURRENT_SOURCE_DIR}/../libcutensor")
set(_default_cutensor_root "${_local_cutensor_root}")
if(NOT EXISTS "${_default_cutensor_root}" AND EXISTS "${_sibling_cutensor_root}")
  set(_default_cutensor_root "${_sibling_cutensor_root}")
endif()
set(CUTENSOR_ROOT "${_default_cutensor_root}" CACHE PATH "Path to cuTENSOR.")

if(NOT EXISTS "${CUTLASS_ROOT}/include/cutlass/cutlass.h" AND EXISTS "${_sibling_cutlass_root}/include/cutlass/cutlass.h")
  set(CUTLASS_ROOT "${_sibling_cutlass_root}" CACHE PATH "Path to CUTLASS." FORCE)
endif()
if(NOT EXISTS "${CUTLASS_ROOT}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "CUTLASS headers not found at ${CUTLASS_ROOT}.")
endif()

if(NOT EXISTS "${CUTENSOR_ROOT}/include/cutensor.h" AND EXISTS "${_sibling_cutensor_root}/include/cutensor.h")
  set(CUTENSOR_ROOT "${_sibling_cutensor_root}" CACHE PATH "Path to cuTENSOR." FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

# GTL: header-only Swiss table (flat_hash_map/set) — Abseil algorithm, standalone
include(FetchContent)
FetchContent_Declare(
  gtl
  GIT_REPOSITORY https://github.com/greg7mdp/gtl.git
  GIT_TAG        v1.2.0
)
FetchContent_MakeAvailable(gtl)

find_path(CUTENSOR_INCLUDE_DIR cutensor.h HINTS "${CUTENSOR_ROOT}/include" "${CUTENSOR_ROOT}" PATH_SUFFIXES include)
find_library(CUTENSOR_LIBRARY NAMES cutensor HINTS "${CUTENSOR_ROOT}/lib" "${CUTENSOR_ROOT}" PATH_SUFFIXES lib lib64)
find_library(CUTENSORMG_LIBRARY NAMES cutensorMg HINTS "${CUTENSOR_ROOT}/lib" "${CUTENSOR_ROOT}" PATH_SUFFIXES lib lib64)

if(NOT CUTENSOR_INCLUDE_DIR OR NOT CUTENSOR_LIBRARY)
  message(FATAL_ERROR "cuTENSOR not found. Set CUTENSOR_ROOT.")
endif()

get_filename_component(CUTENSOR_LIB_DIR "${CUTENSOR_LIBRARY}" DIRECTORY)

# Compiler warnings (apply to our code only, not CUTLASS/cuTENSOR headers)
set(BSPTC_CXX_WARNINGS -Wall -Wextra -Wshadow -Wconversion -Wno-sign-conversion
    -Wnull-dereference -Wformat=2 -Wno-unknown-pragmas)

# Helper macro for CUDA executables (CUTLASS + cuTENSOR)
macro(add_cuda_executable name source)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUTENSOR_INCLUDE_DIR}
  )
  # CUTLASS headers are SYSTEM to suppress their warnings
  target_include_directories(${name} SYSTEM PRIVATE
    ${CUTLASS_ROOT}/include
    ${CUTLASS_ROOT}/tools/util/include
  )
  target_compile_options(${name} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${BSPTC_CXX_WARNINGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${BSPTC_CXX_WARNINGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
  )
  target_link_libraries(${name} PRIVATE CUDA::cudart ${CUTENSOR_LIBRARY})
  if(CUTENSORMG_LIBRARY)
    target_link_libraries(${name} PRIVATE ${CUTENSORMG_LIBRARY})
  endif()
  set_target_properties(${name} PROPERTIES
    BUILD_RPATH "${CUTENSOR_LIB_DIR}"
    INSTALL_RPATH "${CUTENSOR_LIB_DIR}"
  )
endmacro()

# ==============================================================================
# General Sparse Executables
# ==============================================================================

# ==============================================================================
# Shared STATIC libraries — compiled once, linked into all pipeline targets.
# STATIC (not OBJECT) so that target TU compilation runs in parallel with the
# library build; only the link step waits for the library to be ready.
# ==============================================================================

# Helper: configure include dirs + warnings for a library target
macro(configure_lib_target tgt)
  target_include_directories(${tgt} PRIVATE ${CMAKE_SOURCE_DIR}/include ${CUTENSOR_INCLUDE_DIR})
  target_include_directories(${tgt} SYSTEM PRIVATE
    ${CUTLASS_ROOT}/include ${CUTLASS_ROOT}/tools/util/include)
  target_compile_options(${tgt} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${BSPTC_CXX_WARNINGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
endmacro()

# cuBLAS grouped GEMM backend for the Large tier.
# Separate TU with no CUTLASS headers (same isolation rule as cuTENSOR).
add_library(cublas_gemm_lib STATIC ${CMAKE_SOURCE_DIR}/src/cublas/cublas_grouped_gemm.cu)
target_include_directories(cublas_gemm_lib PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(cublas_gemm_lib PUBLIC CUDA::cublas)
target_compile_options(cublas_gemm_lib PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${BSPTC_CXX_WARNINGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)

# CUTLASS grouped GEMM instantiations — split per tile shape for parallel compilation
add_library(cutlass_gemm_lib STATIC
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f32_128x128.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f32_64x64.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f32_32x32.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f32_64x8.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f32_8x64.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f64_64x64.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f64_32x32.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f64_64x8.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f64_8x64.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f32_128x16.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f32_16x128.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f32_256x32.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f32_32x256.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f64_128x16.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f64_16x128.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f64_256x32.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f64_32x256.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f64_128x128.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f64_64x64_tc.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f64_32x32_tc.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f64_128x16_tc.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f64_16x128_tc.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f64_256x32_tc.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_f64_32x256_tc.cu
    ${CMAKE_SOURCE_DIR}/src/cutlass/gemm_dispatch.cu
)
configure_lib_target(cutlass_gemm_lib)
target_link_libraries(cutlass_gemm_lib PUBLIC cublas_gemm_lib)

# Warp GEMM kernel instantiations (32 template variants × 2 types)
add_library(warp_gemm_lib STATIC ${CMAKE_SOURCE_DIR}/src/warp_gemm_instantiate.cu)
configure_lib_target(warp_gemm_lib)

# Hash tables — gtl Swiss table for block shape map + pair-key dedup.
# Pure C++ (no CUDA), isolated from CUTLASS to avoid umul128 collision.
add_library(hash_tables_lib STATIC ${CMAKE_SOURCE_DIR}/src/hash_tables.cpp)
target_include_directories(hash_tables_lib PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(hash_tables_lib PRIVATE gtl)
target_compile_options(hash_tables_lib PRIVATE ${BSPTC_CXX_WARNINGS})

# Pair matching — uses gtl Swiss table, isolated from CUTLASS to avoid umul128 collision
add_library(pair_matching_lib STATIC ${CMAKE_SOURCE_DIR}/src/pair_matching_instantiate.cu)
target_include_directories(pair_matching_lib PRIVATE ${CMAKE_SOURCE_DIR}/include ${CUTENSOR_INCLUDE_DIR})
target_link_libraries(pair_matching_lib PRIVATE gtl)
target_compile_options(pair_matching_lib PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${BSPTC_CXX_WARNINGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)

# Pipeline template instantiations — split by type for parallel compilation.
# Uses extern template declarations in pipeline.cuh to avoid re-instantiating
# pipeline_setup/execute<float/double> in each consuming TU.
add_library(pipeline_f32_lib STATIC ${CMAKE_SOURCE_DIR}/src/pipeline_instantiate.cu)
configure_lib_target(pipeline_f32_lib)
add_library(pipeline_f64_lib STATIC ${CMAKE_SOURCE_DIR}/src/pipeline_instantiate_f64.cu)
configure_lib_target(pipeline_f64_lib)

# Convenience variable for targets that need the full pipeline
set(PIPELINE_LIBS pipeline_f32_lib pipeline_f64_lib warp_gemm_lib cutlass_gemm_lib cublas_gemm_lib pair_matching_lib hash_tables_lib)

# CLI tool for running general sparse contractions
add_cuda_executable(general_sparse src/general_sparse_main.cu)
target_link_libraries(general_sparse PRIVATE ${PIPELINE_LIBS})

# ==============================================================================
# Tests
# ==============================================================================
enable_testing()

# Pipeline verification tests (CUTLASS + naive dispatch, stride-based + cuTENSOR verification)
add_cuda_executable(test_pipeline_verification tests/test_pipeline_verification.cu)
target_include_directories(test_pipeline_verification PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_pipeline_verification PRIVATE ${PIPELINE_LIBS})
add_test(NAME test_pipeline_verification COMMAND test_pipeline_verification)

# Pipeline implementation benchmark (CUTLASS vs cuTENSOR on real data)
add_cuda_executable(bench_pipeline_implementations tests/bench_pipeline_implementations.cu)
target_link_libraries(bench_pipeline_implementations PRIVATE ${PIPELINE_LIBS})

# Synthetic kernel benchmarks (transpose + accumulation, no tensor files needed)
add_cuda_executable(bench_kernels tests/bench_kernels.cu)
target_link_libraries(bench_kernels PRIVATE hash_tables_lib)

# Per-kernel benchmarks (small and medium GEMM)
add_cuda_executable(bench_small tests/bench_small.cu)
target_link_libraries(bench_small PRIVATE ${PIPELINE_LIBS})
add_cuda_executable(bench_medium tests/bench_medium.cu)
target_link_libraries(bench_medium PRIVATE ${PIPELINE_LIBS})
add_cuda_executable(bench_host_stages tests/bench_host_stages.cu)
add_cuda_executable(bench_cutlass_tiles tests/bench_cutlass_tiles.cu)
target_link_libraries(bench_cutlass_tiles PRIVATE ${PIPELINE_LIBS})

# Random contraction fuzzer + scaling benchmark
add_cuda_executable(fuzz_pipeline tests/fuzz_pipeline.cu)
target_include_directories(fuzz_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(fuzz_pipeline PRIVATE ${PIPELINE_LIBS})

# Standalone hypertile transpose tester + statistics
add_cuda_executable(test_hypertile tests/test_hypertile.cu)
target_link_libraries(test_hypertile PRIVATE ${PIPELINE_LIBS})

# cuTENSOR vs hypertile transpose benchmark
add_cuda_executable(bench_transpose_cutensor tests/bench_transpose_cutensor.cu)
target_link_libraries(bench_transpose_cutensor PRIVATE hash_tables_lib)

# Standalone CUTLASS 8x64 skinny tile correctness + benchmark test
add_cuda_executable(test_cutlass_skinny tests/test_cutlass_skinny.cu)

# ==============================================================================
# OpenMP (must come after all targets are defined)
# ==============================================================================
if(OpenMP_CXX_FOUND)
  # Determine the OpenMP flag for nvcc's host compiler.
  # Prefer CMake-detected flags; fall back per compiler ID.
  if(OpenMP_CUDA_FLAGS)
    set(_omp_xcompiler_flag "${OpenMP_CUDA_FLAGS}")
  elseif(OpenMP_CXX_FLAGS)
    set(_omp_xcompiler_flag "${OpenMP_CXX_FLAGS}")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI")
    set(_omp_xcompiler_flag "-mp")
  else()
    set(_omp_xcompiler_flag "-fopenmp")
  endif()
  foreach(_tgt general_sparse test_pipeline_verification bench_pipeline_implementations bench_kernels bench_small bench_medium bench_host_stages fuzz_pipeline test_hypertile bench_transpose_cutensor bench_cutlass_tiles)
    target_link_libraries(${_tgt} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(${_tgt} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_omp_xcompiler_flag}>)
    # OpenMP::OpenMP_CXX link flags may not propagate when nvcc drives the link step.
    # Pass the raw flag (e.g. -fopenmp / -mp) — works whether g++ or nvcc is the linker.
    target_link_options(${_tgt} PRIVATE ${_omp_xcompiler_flag})
  endforeach()
  # Shared STATIC libraries need OpenMP compile flags too
  foreach(_tgt ${PIPELINE_LIBS})
    target_compile_options(${_tgt} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_omp_xcompiler_flag}>)
  endforeach()
endif()

