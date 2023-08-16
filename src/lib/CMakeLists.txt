project(neo_fft)

add_library(neo_fft INTERFACE)
add_library(neo::fft ALIAS neo_fft)

target_include_directories(neo_fft INTERFACE ${PROJECT_SOURCE_DIR})
target_link_libraries(neo_fft INTERFACE std::mdspan)
target_compile_definitions(neo_fft INTERFACE _CRT_SECURE_NO_WARNINGS=1)

add_executable(neo_fft_tests)
target_sources(neo_fft_tests
    PRIVATE
        "neo/algorithm/allclose_test.cpp"
        "neo/algorithm/copy_test.cpp"

        "neo/container/sparse_matrix_test.cpp"

        "neo/fft/convolution/multiply_elements_add_columns_test.cpp"
        "neo/fft/convolution/overlap_test.cpp"
        "neo/fft/convolution/uniform_partition_test.cpp"
        "neo/fft/convolution/uniform_partitioned_convolver_test.cpp"

        "neo/fft/transform/dft_test.cpp"
        "neo/fft/transform/fftfreq_test.cpp"
        "neo/fft/transform/fft_test.cpp"
        "neo/fft/transform/rfft_test.cpp"
        "neo/fft/transform/stft_test.cpp"
        "neo/fft/transform/twiddle_test.cpp"

        "neo/math/a_weighting_test.cpp"
        "neo/math/complex_test.cpp"
        "neo/math/fixed_point_test.cpp"
        "neo/math/decibel_test.cpp"
        "neo/math/ilog2_test.cpp"
        "neo/math/next_power_of_two_test.cpp"
        "neo/math/power_test.cpp"
        "neo/math/windowing_test.cpp"

        "neo/simd_test.cpp"
)

catch_discover_tests(neo_fft_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
target_link_libraries(neo_fft_tests
    PRIVATE
        neo::fft
        Catch2::Catch2WithMain
        juce::juce_recommended_warning_flags
)
