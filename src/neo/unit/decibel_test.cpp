// SPDX-License-Identifier: MIT

#include "decibel.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEMPLATE_TEST_CASE("neo/math: amplitude_to_db<precision::accurate>", "", float, double)
{
    using Float = TestType;

    REQUIRE(neo::amplitude_to_db<neo::precision::accurate>(Float(1.0)) == Catch::Approx(0.0));
    REQUIRE(neo::amplitude_to_db<neo::precision::accurate>(Float(0.5)) == Catch::Approx(-6.020599));
    REQUIRE(neo::amplitude_to_db<neo::precision::accurate>(Float(0.25)) == Catch::Approx(-12.04119));
    REQUIRE(neo::amplitude_to_db<neo::precision::accurate>(Float(0.125)) == Catch::Approx(-18.061799));

    REQUIRE(neo::amplitude_to_db<neo::precision::accurate>(Float(0)) == Catch::Approx(-144.0));
    REQUIRE(neo::amplitude_to_db<neo::precision::accurate>(Float(0), Float(-100.0)) == Catch::Approx(-100.0));
    REQUIRE(neo::amplitude_to_db<neo::precision::accurate>(Float(0), Float(-50.0)) == Catch::Approx(-50.0));
    REQUIRE(neo::amplitude_to_db<neo::precision::accurate>(Float(0.00001), Float(-50.0)) == Catch::Approx(-50.0));
}

TEMPLATE_TEST_CASE("neo/math: amplitude_to_db<precision::estimate>", "", float, double)
{
    using Float = TestType;

    REQUIRE(neo::amplitude_to_db<neo::precision::estimate>(Float(0.5)) == Catch::Approx(-6.020599));
    REQUIRE(neo::amplitude_to_db<neo::precision::estimate>(Float(0.25)) == Catch::Approx(-12.04119));
    REQUIRE(neo::amplitude_to_db<neo::precision::estimate>(Float(0.125)) == Catch::Approx(-18.061799));

    REQUIRE(neo::amplitude_to_db<neo::precision::estimate>(Float(0)) == Catch::Approx(-144.0));
    REQUIRE(neo::amplitude_to_db<neo::precision::estimate>(Float(0), Float(-100.0)) == Catch::Approx(-100.0));
    REQUIRE(neo::amplitude_to_db<neo::precision::estimate>(Float(0), Float(-50.0)) == Catch::Approx(-50.0));
    REQUIRE(neo::amplitude_to_db<neo::precision::estimate>(Float(0.00001), Float(-50.0)) == Catch::Approx(-50.0));
}
