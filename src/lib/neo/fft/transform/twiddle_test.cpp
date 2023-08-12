#include "twiddle.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/fft/transform: make_radix2_twiddles", "", float, double, long double)
{
    using Float   = TestType;
    using Complex = std::complex<Float>;

    auto const array  = neo::fft::make_radix2_twiddles<Complex, 64>();
    auto const vector = neo::fft::make_radix2_twiddles<Complex>(64);

    for (auto i{0UL}; i < array.size(); ++i) {
        CAPTURE(i);
        REQUIRE(array(i).real() == Catch::Approx(vector(i).real()));
        REQUIRE(array(i).imag() == Catch::Approx(vector(i).imag()));
    }
}
