#include "decibel.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEMPLATE_TEST_CASE("neo/math: to_decibels", "", float, double, long double)
{
    using Float = TestType;

    auto const precision = GENERATE(neo::math_precision::accurate, neo::math_precision::estimate);

    REQUIRE(neo::to_decibels(Float(0.5), precision) == Catch::Approx(-6.020599));
    REQUIRE(neo::to_decibels(Float(0.25), precision) == Catch::Approx(-12.04119));
    REQUIRE(neo::to_decibels(Float(0.125), precision) == Catch::Approx(-18.061799));

    REQUIRE(neo::to_decibels(Float(0), precision) == Catch::Approx(-144.0));
    REQUIRE(neo::to_decibels(Float(0), Float(-100.0), precision) == Catch::Approx(-100.0));
    REQUIRE(neo::to_decibels(Float(0), Float(-50.0), precision) == Catch::Approx(-50.0));
    REQUIRE(neo::to_decibels(Float(0.00001), Float(-50.0), precision) == Catch::Approx(-50.0));

    REQUIRE(neo::to_decibels(Float(1), neo::math_precision::accurate) == Catch::Approx(0.0));
}
