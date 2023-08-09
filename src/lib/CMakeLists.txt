project(neo_convolution)

add_library(neo_fft INTERFACE)
add_library(neo::fft ALIAS neo_fft)
target_include_directories(neo_fft INTERFACE ${PROJECT_SOURCE_DIR})
target_compile_definitions(neo_fft INTERFACE _CRT_SECURE_NO_WARNINGS=1)

add_library(neo_wav neo/wav/wav.cpp neo/wav/wav.hpp)
add_library(neo::wav ALIAS neo_wav)
target_include_directories(neo_wav INTERFACE ${PROJECT_SOURCE_DIR})
target_compile_definitions(neo_wav INTERFACE _CRT_SECURE_NO_WARNINGS=1)

add_library(neo_convolution)
add_library(neo::convolution ALIAS neo_convolution)

target_include_directories(neo_convolution PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(neo_convolution PUBLIC neo::fft std::mdspan)

set(neo_convolution_Sources
    "neo/convolution.hpp"
    "neo/convolution.cpp"

    "neo/convolution/container.hpp"
    "neo/convolution/container/mdspan.hpp"
    "neo/convolution/container/sparse_matrix.hpp"

    "neo/convolution/convolver.hpp"
    "neo/convolution/convolver/sparse_upols_convolver.cpp"
    "neo/convolution/convolver/sparse_upols_convolver.hpp"
    "neo/convolution/convolver/upols_convolver.cpp"
    "neo/convolution/convolver/upols_convolver.hpp"

    "neo/convolution/math/fixed_point.hpp"
    "neo/convolution/math/fixed_point/algorithm.hpp"
    "neo/convolution/math/fixed_point/complex.hpp"
    "neo/convolution/math/fixed_point/fixed_point.hpp"

    "neo/convolution/math/magnitude_scaling.hpp"
    "neo/convolution/math/magnitude_scaling/a_weighting.hpp"
)

target_sources(neo_convolution PRIVATE ${neo_convolution_Sources})
set_target_properties(neo_convolution PROPERTIES UNITY_BUILD ON UNITY_BUILD_MODE GROUP)
set_source_files_properties(neo_convolution ${neo_convolution_Sources} PROPERTIES UNITY_GROUP "src")
