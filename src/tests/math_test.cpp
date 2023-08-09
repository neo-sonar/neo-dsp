#include "neo/fft/math.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/fft/math: a_weighting", "", float, double)
{
    using Float = TestType;

    REQUIRE(neo::fft::a_weighting(Float(24.5)) == Catch::Approx(Float(-45.30166390)));     // G0
    REQUIRE(neo::fft::a_weighting(Float(49.0)) == Catch::Approx(Float(-30.64262470)));     // G1
    REQUIRE(neo::fft::a_weighting(Float(98.0)) == Catch::Approx(Float(-19.42442872)));     // G2
    REQUIRE(neo::fft::a_weighting(Float(196.0)) == Catch::Approx(Float(-11.05317378)));    // G3
    REQUIRE(neo::fft::a_weighting(Float(392.0)) == Catch::Approx(Float(-4.92218165)));     // G4
    REQUIRE(neo::fft::a_weighting(Float(783.99)) == Catch::Approx(Float(-0.87787452)));    // G5
    REQUIRE(neo::fft::a_weighting(Float(1567.98)) == Catch::Approx(Float(0.96689509)));    // G6
    REQUIRE(neo::fft::a_weighting(Float(3135.96)) == Catch::Approx(Float(1.20425069)));    // G7
    REQUIRE(neo::fft::a_weighting(Float(6271.93)) == Catch::Approx(Float(-0.09973374)));   // G8
    REQUIRE(neo::fft::a_weighting(Float(12543.85)) == Catch::Approx(Float(-4.28495301)));  // G9
}

TEMPLATE_TEST_CASE("neo/fft/math: decibel", "", float, double)
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

TEMPLATE_TEST_CASE("neo/fft/math: next_power_of_two", "", std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t)
{
    using UInt = TestType;

    REQUIRE(neo::fft::next_power_of_two(UInt(1)) == UInt(1));
    REQUIRE(neo::fft::next_power_of_two(UInt(2)) == UInt(2));
    REQUIRE(neo::fft::next_power_of_two(UInt(3)) == UInt(4));
    REQUIRE(neo::fft::next_power_of_two(UInt(4)) == UInt(4));
    REQUIRE(neo::fft::next_power_of_two(UInt(100)) == UInt(128));
}

TEMPLATE_TEST_CASE(
    "neo/fft/math: detail::next_power_of_two_fallback",
    "",
    std::uint8_t,
    std::uint16_t,
    std::uint32_t,
    std::uint64_t
)
{
    using UInt = TestType;

    REQUIRE(neo::fft::detail::next_power_of_two_fallback(UInt(1)) == UInt(1));
    REQUIRE(neo::fft::detail::next_power_of_two_fallback(UInt(2)) == UInt(2));
    REQUIRE(neo::fft::detail::next_power_of_two_fallback(UInt(3)) == UInt(4));
    REQUIRE(neo::fft::detail::next_power_of_two_fallback(UInt(4)) == UInt(4));
    REQUIRE(neo::fft::detail::next_power_of_two_fallback(UInt(100)) == UInt(128));
}
