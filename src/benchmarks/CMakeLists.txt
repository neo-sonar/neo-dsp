project(neo_convolution_benchmarks)

add_executable(fixed_point_benchmark fixed_point.cpp)
target_link_libraries(fixed_point_benchmark PRIVATE neo::convolution)

