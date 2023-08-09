project(neo_convolution)

add_library(neo_convolution)
add_library(neo::convolution ALIAS neo_convolution)

set(neo_convolution_Sources
    "neo/fft/convolution.hpp"
    "neo/fft/math.hpp"
    "neo/fft/transform.hpp"

    "neo/fft/convolution/container.hpp"
    "neo/fft/convolution/container/mdspan.hpp"
    "neo/fft/convolution/container/sparse_matrix.hpp"

    "neo/fft/convolution/convolver.hpp"
    "neo/fft/convolution/convolver/sparse_upols_convolver.cpp"
    "neo/fft/convolution/convolver/sparse_upols_convolver.hpp"
    "neo/fft/convolution/convolver/upols_convolver.cpp"
    "neo/fft/convolution/convolver/upols_convolver.hpp"

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

target_sources(neo_convolution PRIVATE ${neo_convolution_Sources})
set_target_properties(neo_convolution PROPERTIES UNITY_BUILD ON UNITY_BUILD_MODE GROUP)
set_source_files_properties(neo_convolution ${neo_convolution_Sources} PROPERTIES UNITY_GROUP "src")

target_include_directories(neo_convolution PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(neo_convolution PUBLIC std::mdspan)
target_compile_definitions(neo_convolution INTERFACE _CRT_SECURE_NO_WARNINGS=1)

add_library(neo_wav neo/wav/wav.cpp neo/wav/wav.hpp)
add_library(neo::wav ALIAS neo_wav)
target_include_directories(neo_wav INTERFACE ${PROJECT_SOURCE_DIR})
target_compile_definitions(neo_wav INTERFACE _CRT_SECURE_NO_WARNINGS=1)
