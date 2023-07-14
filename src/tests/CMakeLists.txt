project(neo_convolution_tests)

add_executable(fixed_point_test fixed_point.t.cpp)
add_test(NAME fixed_point_test COMMAND fixed_point_test)
target_link_libraries(fixed_point_test PRIVATE neo::convolution)

add_executable(sparse_matrix_test sparse_matrix.t.cpp)
add_test(NAME sparse_matrix_test COMMAND sparse_matrix_test)
target_link_libraries(sparse_matrix_test PRIVATE neo::convolution)
