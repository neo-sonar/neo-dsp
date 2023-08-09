project(neo_fft)

add_library(neo_fft)
add_library(neo::convolution ALIAS neo_fft)

set(neo_fft_Sources
    "neo/fft/container.hpp"
    "neo/fft/convolution.hpp"
    "neo/fft/math.hpp"
    "neo/fft/transform.hpp"

    "neo/fft/container/mdspan.hpp"
    "neo/fft/container/sparse_matrix.hpp"

    "neo/fft/convolution.hpp"
    "neo/fft/convolution/sparse_upols_convolver.cpp"
    "neo/fft/convolution/sparse_upols_convolver.hpp"
    "neo/fft/convolution/upols_convolver.cpp"
    "neo/fft/convolution/upols_convolver.hpp"

    "neo/fft/math/fixed_point.hpp"
    "neo/fft/math/fixed_point/algorithm.hpp"
    "neo/fft/math/fixed_point/complex.hpp"
    "neo/fft/math/fixed_point/fixed_point.hpp"

    "neo/fft/math/a_weighting.hpp"
    "neo/fft/math/decibel.hpp"
    "neo/fft/math/divide_round_up.hpp"
    "neo/fft/math/ilog2.hpp"
    "neo/fft/math/next_power_of_two.hpp"
    "neo/fft/math/power.hpp"
)

target_sources(neo_fft PRIVATE ${neo_fft_Sources})
set_target_properties(neo_fft PROPERTIES UNITY_BUILD ON UNITY_BUILD_MODE GROUP)
set_source_files_properties(neo_fft ${neo_fft_Sources} PROPERTIES UNITY_GROUP "src")

target_include_directories(neo_fft PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(neo_fft PUBLIC std::mdspan)
target_compile_definitions(neo_fft INTERFACE _CRT_SECURE_NO_WARNINGS=1)
