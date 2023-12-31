project(neo_benchmarks)

FetchContent_Declare(benchmark GIT_REPOSITORY "https://github.com/google/benchmark" GIT_TAG "v1.8.3" GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(benchmark)

add_executable(fixed_point_benchmark src/fixed_point.cpp)
target_link_libraries(fixed_point_benchmark PRIVATE neosonar::neo fmt::fmt-header-only)

add_executable(convolution_benchmark src/convolution.cpp)
target_link_libraries(convolution_benchmark PRIVATE neosonar::neo benchmark::benchmark)

add_executable(fft_benchmark src/fft.cpp)
target_link_libraries(fft_benchmark PRIVATE neosonar::neo benchmark::benchmark)

add_executable(multiply_add_benchmark src/multiply_add.cpp)
target_link_libraries(multiply_add_benchmark PRIVATE neosonar::neo benchmark::benchmark)

add_executable(rfft_benchmark src/rfft.cpp)
target_link_libraries(rfft_benchmark PRIVATE neosonar::neo benchmark::benchmark)

add_executable(split_complex_benchmark src/split_complex.cpp)
target_link_libraries(split_complex_benchmark PRIVATE neosonar::neo benchmark::benchmark)
