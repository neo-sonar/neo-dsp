project(neo_convolution_tests)

add_executable(neo_convolution_tests)
target_sources(neo_convolution_tests
    PRIVATE
        test.cpp
        a_weighting_test.cpp
        fixed_point_test.cpp
        sparse_matrix_test.cpp
)

catch_discover_tests(neo_convolution_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
target_link_libraries(neo_convolution_tests
    PRIVATE
        neo::convolution
        Catch2::Catch2WithMain
        juce::juce_recommended_warning_flags
)

add_executable(neo_fft_wav wav.cpp)
target_link_libraries(neo_fft_wav
    PRIVATE
        neo::convolution
        neo::wav juce::juce_recommended_warning_flags
)
