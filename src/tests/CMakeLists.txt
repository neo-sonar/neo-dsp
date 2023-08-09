project(neo_fft_tests)

add_executable(neo_fft_tests)
target_sources(neo_fft_tests
    PRIVATE
        test.cpp
        fixed_point_test.cpp
        math_test.cpp
        sparse_matrix_test.cpp
)

catch_discover_tests(neo_fft_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
target_link_libraries(neo_fft_tests
    PRIVATE
        neo::convolution
        Catch2::Catch2WithMain
        juce::juce_recommended_warning_flags
)
