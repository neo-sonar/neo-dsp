project(neo_convolution)

add_library(neo_convolution)
add_library(neo::convolution ALIAS neo_convolution)

target_include_directories(neo_convolution PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(neo_convolution PUBLIC neo::fft std::mdspan)

target_sources(neo_convolution
    PUBLIC
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
