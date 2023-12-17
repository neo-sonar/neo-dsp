project(neo_dsp_benchmarks)

FetchContent_Declare(benchmark GIT_REPOSITORY "https://github.com/google/benchmark" GIT_TAG "v1.8.3" GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(benchmark)

add_executable(fixed_point_benchmark fixed_point.cpp)
target_link_libraries(fixed_point_benchmark PRIVATE neo::dsp fmt::fmt-header-only)

add_executable(fft_benchmark fft.cpp)
target_link_libraries(fft_benchmark PRIVATE neo::dsp fmt::fmt-header-only)
if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    target_link_libraries(fft_benchmark PRIVATE ${ACCELERATE_LIBRARY})
endif()

add_executable(split_complex_benchmark split_complex.cpp)
target_link_libraries(split_complex_benchmark PRIVATE neo::dsp fmt::fmt-header-only)
if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    target_link_libraries(split_complex_benchmark PRIVATE ${ACCELERATE_LIBRARY})
endif()

add_executable(rfft_benchmark rfft.cpp)
target_link_libraries(rfft_benchmark PRIVATE neo::dsp fmt::fmt-header-only benchmark::benchmark)
if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    target_link_libraries(rfft_benchmark PRIVATE ${ACCELERATE_LIBRARY})
endif()
