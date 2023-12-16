#include "twiddle.hpp"

#include <neo/complex/scalar_complex.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEMPLATE_PRODUCT_TEST_CASE("neo/fft: make_radix2_twiddles", "", (std::complex, neo::scalar_complex), (float, double))
{
    using Complex = TestType;

    auto const array  = neo::fft::make_radix2_twiddles<Complex, 8>();
    auto const vector = neo::fft::make_radix2_twiddles<Complex>(8);

    auto const margin = 0.000001;
    REQUIRE_THAT(vector(0).real(), Catch::Matchers::WithinAbs(1.0, margin));
    REQUIRE_THAT(vector(0).imag(), Catch::Matchers::WithinAbs(0.0, margin));

    REQUIRE_THAT(vector(1).real(), Catch::Matchers::WithinAbs(0.707107, margin));
    REQUIRE_THAT(vector(1).imag(), Catch::Matchers::WithinAbs(-0.707107, margin));

    REQUIRE_THAT(vector(2).real(), Catch::Matchers::WithinAbs(0.0, margin));
    REQUIRE_THAT(vector(2).imag(), Catch::Matchers::WithinAbs(-1.0, margin));

    REQUIRE_THAT(vector(3).real(), Catch::Matchers::WithinAbs(-0.707107, margin));
    REQUIRE_THAT(vector(3).imag(), Catch::Matchers::WithinAbs(-0.707107, margin));

    for (auto i{0UL}; i < array.size(); ++i) {
        CAPTURE(i);
        REQUIRE(array(i).real() == Catch::Approx(vector(i).real()));
        REQUIRE(array(i).imag() == Catch::Approx(vector(i).imag()));
    }
}
