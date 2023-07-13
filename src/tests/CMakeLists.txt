project(neo_convolution_tests)

add_executable(sparse_matrix_test sparse_matrix.t.cpp)
add_test(NAME sparse_matrix_test COMMAND sparse_matrix_test)
target_link_libraries(sparse_matrix_test PRIVATE neo::convolution)