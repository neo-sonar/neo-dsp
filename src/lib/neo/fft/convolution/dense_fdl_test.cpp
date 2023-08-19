#include "dense_fdl.hpp"

#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE(
    "neo/fft/convolution: dense_fdl",
    "",
    std::complex<float>,
    std::complex<double>,
    neo::complex64,
    neo::complex128
)
{
    using Complex = TestType;
    using Fdl     = neo::fft::dense_fdl<Complex>;
    STATIC_REQUIRE(std::same_as<typename Fdl::value_type, Complex>);
}
