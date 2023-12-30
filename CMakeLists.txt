cmake_minimum_required(VERSION 3.23...3.27)
project(neo-dsp-dev VERSION 0.1.0)

option(NEO_ENABLE_APPLE_VDSP "Link against Apple vDSP" ON)
option(NEO_ENABLE_INTEL_IPP "Link against Intel IPP" OFF)
option(NEO_ENABLE_INTEL_MKL "Link against Intel MKL" OFF)
option(NEO_ENABLE_XSIMD "Link against xsimd" ON)

find_program(CCACHE ccache)
if (CCACHE)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
endif ()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()
include(FetchContent)

FetchContent_Declare(mdspan GIT_REPOSITORY "https://github.com/kokkos/mdspan" GIT_TAG "stable" GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(mdspan)

if(NEO_ENABLE_XSIMD)
    FetchContent_Declare(xsimd GIT_REPOSITORY "https://github.com/xtensor-stack/xsimd" GIT_TAG "12.1.1" GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(xsimd)
endif()

add_subdirectory(src)

if(DEFINED SKBUILD)
    add_subdirectory(python)
    return()
endif()

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Enable Google benchmark tests")
FetchContent_Declare(fmt GIT_REPOSITORY "https://github.com/fmtlib/fmt" GIT_TAG "10.1.1" GIT_SHALLOW TRUE)
FetchContent_Declare(Catch2 GIT_REPOSITORY "https://github.com/catchorg/Catch2" GIT_TAG "devel" GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(fmt Catch2)
include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)

add_executable(neo_dsp_tests)
target_sources(neo_dsp_tests
    PRIVATE
        "src/neo/algorithm/allclose_test.cpp"
        "src/neo/algorithm/allmatch_test.cpp"
        "src/neo/algorithm/copy_test.cpp"
        "src/neo/algorithm/mean_squared_error_test.cpp"
        "src/neo/algorithm/mean_test.cpp"
        "src/neo/algorithm/multiply_add_test.cpp"
        "src/neo/algorithm/normalize_energy_test.cpp"
        "src/neo/algorithm/standard_deviation_test.cpp"
        "src/neo/algorithm/variance_test.cpp"

        "src/neo/complex/complex_test.cpp"
        "src/neo/complex/interleave_complex_test.cpp"
        "src/neo/complex/parallel_complex_test.cpp"
        "src/neo/complex/scalar_complex_test.cpp"
        "src/neo/complex/split_complex_test.cpp"

        "src/neo/container/compressed_accessor_test.cpp"
        "src/neo/container/csr_matrix_test.cpp"

        "src/neo/convolution/compressed_fdl_test.cpp"
        "src/neo/convolution/dense_fdl_test.cpp"
        "src/neo/convolution/direct_convolve_test.cpp"
        "src/neo/convolution/fdl_index_test.cpp"
        "src/neo/convolution/fft_convolver_test.cpp"
        "src/neo/convolution/normalize_impulse_test.cpp"
        "src/neo/convolution/overlap_test.cpp"
        "src/neo/convolution/uniform_partition_test.cpp"
        "src/neo/convolution/uniform_partitioned_convolver_test.cpp"

        "src/neo/fft/dct_test.cpp"
        "src/neo/fft/dft_test.cpp"
        "src/neo/fft/rfftfreq_test.cpp"
        "src/neo/fft/fft_test.cpp"
        "src/neo/fft/rfft_test.cpp"
        "src/neo/fft/stft_test.cpp"

        "src/neo/fixed_point/fixed_point_test.cpp"

        "src/neo/math/a_weighting_test.cpp"
        "src/neo/math/bit_ceil_test.cpp"
        "src/neo/math/ilog2_test.cpp"
        "src/neo/math/ipow_test.cpp"
        "src/neo/math/windowing_test.cpp"

        "src/neo/simd_test.cpp"

        "src/neo/unit/decibel_test.cpp"
        "src/neo/unit/mel_test.cpp"
)

catch_discover_tests(neo_dsp_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

target_link_libraries(neo_dsp_tests
    PRIVATE
        neo::dsp
        Catch2::Catch2WithMain
        juce::juce_recommended_warning_flags
)

add_subdirectory(tool/benchmark)
add_subdirectory(tool/cli)
add_subdirectory(tool/plugin)
