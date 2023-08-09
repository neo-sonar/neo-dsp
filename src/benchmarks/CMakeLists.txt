project(neo_convolution_benchmarks)

add_executable(fixed_point_benchmark fixed_point.cpp)
target_link_libraries(fixed_point_benchmark
    PRIVATE
        neo::convolution
        juce::juce_recommended_warning_flags
)

add_executable(neo_fft_benchmarks fft.cpp)
target_link_libraries(neo_fft_benchmarks
    PRIVATE
        neo::convolution
        juce::juce_recommended_warning_flags
)
