// SPDX-License-Identifier: MIT

#include "a_weighting.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/math: a_weighting", "", float, double)
{
    using Float = TestType;

    REQUIRE(neo::a_weighting(Float(24.5)) == Catch::Approx(-45.30166390).margin(0.0001));     // G0
    REQUIRE(neo::a_weighting(Float(49.0)) == Catch::Approx(-30.64262470).margin(0.0001));     // G1
    REQUIRE(neo::a_weighting(Float(98.0)) == Catch::Approx(-19.42442872).margin(0.0001));     // G2
    REQUIRE(neo::a_weighting(Float(196.0)) == Catch::Approx(-11.05317378).margin(0.0001));    // G3
    REQUIRE(neo::a_weighting(Float(392.0)) == Catch::Approx(-4.92218165).margin(0.0001));     // G4
    REQUIRE(neo::a_weighting(Float(783.99)) == Catch::Approx(-0.87787452).margin(0.0001));    // G5
    REQUIRE(neo::a_weighting(Float(1567.98)) == Catch::Approx(0.96689509).margin(0.0001));    // G6
    REQUIRE(neo::a_weighting(Float(3135.96)) == Catch::Approx(1.20425069).margin(0.0001));    // G7
    REQUIRE(neo::a_weighting(Float(6271.93)) == Catch::Approx(-0.09973374).margin(0.0001));   // G8
    REQUIRE(neo::a_weighting(Float(12543.85)) == Catch::Approx(-4.28495301).margin(0.0001));  // G9
}
