project(neo_fft_benchmarks)

add_executable(fixed_point_benchmark fixed_point.cpp)
target_link_libraries(fixed_point_benchmark PRIVATE neo::fft)

add_executable(fft_benchmark fft.cpp)
target_link_libraries(fft_benchmark PRIVATE neo::fft)
if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    target_link_libraries(fft_benchmark PRIVATE ${ACCELERATE_LIBRARY})
endif()
