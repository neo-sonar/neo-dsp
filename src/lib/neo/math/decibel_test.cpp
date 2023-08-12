#include "decibel.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/math: decibel", "", float, double)
{
    using Float = TestType;

    REQUIRE(neo::fft::to_decibels(Float(1)) == Catch::Approx(0.0));
    REQUIRE(neo::fft::to_decibels(Float(0.5)) == Catch::Approx(-6.020599));
    REQUIRE(neo::fft::to_decibels(Float(0.25)) == Catch::Approx(-12.04119));
    REQUIRE(neo::fft::to_decibels(Float(0.125)) == Catch::Approx(-18.061799));

    REQUIRE(neo::fft::to_decibels(Float(0)) == Catch::Approx(-144.0));
    REQUIRE(neo::fft::to_decibels(Float(0), Float(-100.0)) == Catch::Approx(-100.0));
    REQUIRE(neo::fft::to_decibels(Float(0), Float(-50.0)) == Catch::Approx(-50.0));
    REQUIRE(neo::fft::to_decibels(Float(0.00001), Float(-50.0)) == Catch::Approx(-50.0));
}
