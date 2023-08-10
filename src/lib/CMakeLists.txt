project(neo_fft)

add_library(neo_fft)
add_library(neo::convolution ALIAS neo_fft)

set(neo_fft_Sources
    "neo/fft/algorithm.hpp"
    "neo/fft/algorithm/allclose.hpp"
    "neo/fft/algorithm/rms_error.hpp"

    "neo/fft/container.hpp"
    "neo/fft/container/mdspan.hpp"
    "neo/fft/container/sparse_matrix.hpp"

    "neo/fft/convolution.hpp"
    "neo/fft/convolution/overlap_save.hpp"
    "neo/fft/convolution/sparse_upols_convolver.cpp"
    "neo/fft/convolution/sparse_upols_convolver.hpp"
    "neo/fft/convolution/upols_convolver.hpp"

    "neo/fft/fixed_point.hpp"
    "neo/fft/fixed_point/algorithm.hpp"
    "neo/fft/fixed_point/complex.hpp"
    "neo/fft/fixed_point/fixed_point.hpp"

    "neo/fft/math.hpp"
    "neo/fft/math/a_weighting.hpp"
    "neo/fft/math/decibel.hpp"
    "neo/fft/math/divide_round_up.hpp"
    "neo/fft/math/ilog2.hpp"
    "neo/fft/math/next_power_of_two.hpp"
    "neo/fft/math/power.hpp"

    "neo/fft/transform.hpp"
    "neo/fft/transform/conjugate_view.hpp"
    "neo/fft/transform/dft.hpp"
    "neo/fft/transform/direction.hpp"
    "neo/fft/transform/radix2.hpp"
    "neo/fft/transform/reorder.hpp"
    "neo/fft/transform/rfft.hpp"
    "neo/fft/transform/simd.hpp"
)

target_sources(neo_fft PRIVATE ${neo_fft_Sources})
set_target_properties(neo_fft PROPERTIES UNITY_BUILD ON UNITY_BUILD_MODE GROUP)
set_source_files_properties(neo_fft ${neo_fft_Sources} PROPERTIES UNITY_GROUP "src")

target_include_directories(neo_fft PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(neo_fft PUBLIC std::mdspan)
target_compile_definitions(neo_fft INTERFACE _CRT_SECURE_NO_WARNINGS=1)

add_executable(neo_fft_tests)
target_sources(neo_fft_tests
    PRIVATE
        "neo/fft/math_test.cpp"
        "neo/fft/algorithm/multiply_elementwise_sum_columnwise_test.cpp"
        "neo/fft/container/sparse_matrix_test.cpp"
        "neo/fft/convolution/overlap_save_test.cpp"
        "neo/fft/convolution/sparse_upols_convolver_test.cpp"
        "neo/fft/convolution/upols_convolver_test.cpp"
        "neo/fft/convolution/uniform_partition_test.cpp"
        "neo/fft/fixed_point/fixed_point_test.cpp"
        "neo/fft/transform/radix2_test.cpp"
        "neo/fft/transform/rfft_test.cpp"
)

catch_discover_tests(neo_fft_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
target_link_libraries(neo_fft_tests
    PRIVATE
        neo::convolution
        Catch2::Catch2WithMain
        juce::juce_recommended_warning_flags
)
