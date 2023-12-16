project(neo_fft)

add_library(neo_fft INTERFACE)
add_library(neo::fft ALIAS neo_fft)

target_include_directories(neo_fft INTERFACE ${PROJECT_SOURCE_DIR})
target_link_libraries(neo_fft INTERFACE std::mdspan)
target_compile_definitions(neo_fft INTERFACE _CRT_SECURE_NO_WARNINGS=1)

# add_executable(neo_fft_tests)
# target_sources(neo_fft_tests
#     PRIVATE
#         "neo/algorithm/allclose_test.cpp"
#         "neo/algorithm/allmatch_test.cpp"
#         "neo/algorithm/copy_test.cpp"
#         "neo/algorithm/mean_squared_error_test.cpp"
#         "neo/algorithm/mean_test.cpp"
#         "neo/algorithm/multiply_add_test.cpp"
#         "neo/algorithm/normalize_energy_test.cpp"
#         "neo/algorithm/standard_deviation_test.cpp"
#         "neo/algorithm/variance_test.cpp"

#         "neo/complex/complex_test.cpp"
#         "neo/complex/interleave_complex_test.cpp"
#         "neo/complex/parallel_complex_test.cpp"
#         "neo/complex/scalar_complex_test.cpp"
#         "neo/complex/split_complex_test.cpp"

#         "neo/container/compressed_accessor_test.cpp"
#         "neo/container/sparse_matrix_test.cpp"

#         "neo/convolution/compressed_fdl_test.cpp"
#         "neo/convolution/dense_fdl_test.cpp"
#         "neo/convolution/fdl_index_test.cpp"
#         "neo/convolution/normalize_impulse_test.cpp"
#         "neo/convolution/overlap_test.cpp"
#         "neo/convolution/uniform_partition_test.cpp"
#         "neo/convolution/uniform_partitioned_convolver_test.cpp"

#         "neo/fft/bluestein_test.cpp"
#         "neo/fft/dct_test.cpp"
#         "neo/fft/dft_test.cpp"
#         "neo/fft/fftfreq_test.cpp"
#         "neo/fft/fft_test.cpp"
#         "neo/fft/rfft_test.cpp"
#         "neo/fft/static_fft_test.cpp"
#         "neo/fft/stft_test.cpp"
#         "neo/fft/twiddle_test.cpp"

#         "neo/fixed_point/fixed_point_test.cpp"

#         "neo/math/a_weighting_test.cpp"
#         "neo/math/bit_ceil_test.cpp"
#         "neo/math/ilog2_test.cpp"
#         "neo/math/ipow_test.cpp"
#         "neo/math/windowing_test.cpp"

#         "neo/simd_test.cpp"

#         "neo/unit/decibel_test.cpp"
#         "neo/unit/mel_test.cpp"
# )

# catch_discover_tests(neo_fft_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# target_link_libraries(neo_fft_tests
#     PRIVATE
#         neo::fft
#         Catch2::Catch2WithMain
#         juce::juce_recommended_warning_flags
# )

# if(APPLE)
#     find_library(ACCELERATE_LIBRARY Accelerate)
#     target_link_libraries(neo_fft_tests PRIVATE ${ACCELERATE_LIBRARY})
# endif()
