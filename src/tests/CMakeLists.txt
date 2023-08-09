project(neo_convolution_tests)

add_executable(a_weighting_test a_weighting_test.cpp)
add_test(NAME a_weighting_test COMMAND a_weighting_test)
target_link_libraries(a_weighting_test PRIVATE neo::convolution)

add_executable(fixed_point_test fixed_point_test.cpp)
add_test(NAME fixed_point_test COMMAND fixed_point_test)
target_link_libraries(fixed_point_test PRIVATE neo::convolution)

add_executable(sparse_matrix_test sparse_matrix_test.cpp)
add_test(NAME sparse_matrix_test COMMAND sparse_matrix_test)
target_link_libraries(sparse_matrix_test PRIVATE neo::convolution)

add_executable(neo_fft_tests test.cpp)
add_test(NAME neo_fft_tests COMMAND neo_fft_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
target_link_libraries(neo_fft_tests PRIVATE neo::convolution Catch2::Catch2WithMain)
if(NOT MSVC)
    target_compile_options(neo_fft_tests INTERFACE "-Wall" "-Wextra" "-Wpedantic")
endif()

add_executable(neo_fft_benchmarks benchmark.cpp)
target_link_libraries(neo_fft_benchmarks PRIVATE neo::convolution)
if(NOT MSVC)
    target_compile_options(neo_fft_benchmarks INTERFACE "-Wall" "-Wextra" "-Wpedantic")
endif()

add_executable(neo_fft_wav wav.cpp)
target_link_libraries(neo_fft_wav PRIVATE neo::convolution neo::wav)
if(NOT MSVC)
    target_compile_options(neo_fft_wav INTERFACE "-Wall" "-Wextra" "-Wpedantic")
endif()
