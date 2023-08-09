#include "catch2/catch_test_macros.hpp"
#include "neo/fft/math/a_weighting.hpp"

#include <algorithm>

template<std::floating_point Float>
[[nodiscard]] static auto approx_equal(Float a, Float b) -> bool
{
    if constexpr (std::same_as<Float, float>) {
        auto const tolerance = 0.0001;
        return std::abs(a - b) < tolerance;
    } else {
        auto const tolerance = 0.000001;
        return std::abs(a - b) < tolerance;
    }
}

template<std::floating_point Float>
[[nodiscard]] static auto test_a_weighting() -> bool
{
    REQUIRE(approx_equal(neo::fft::a_weighting(Float(24.5)), Float(-45.30166390)));     // G0
    REQUIRE(approx_equal(neo::fft::a_weighting(Float(49.0)), Float(-30.64262470)));     // G1
    REQUIRE(approx_equal(neo::fft::a_weighting(Float(98.0)), Float(-19.42442872)));     // G2
    REQUIRE(approx_equal(neo::fft::a_weighting(Float(196.0)), Float(-11.05317378)));    // G3
    REQUIRE(approx_equal(neo::fft::a_weighting(Float(392.0)), Float(-4.92218165)));     // G4
    REQUIRE(approx_equal(neo::fft::a_weighting(Float(783.99)), Float(-0.87787452)));    // G5
    REQUIRE(approx_equal(neo::fft::a_weighting(Float(1567.98)), Float(0.96689509)));    // G6
    REQUIRE(approx_equal(neo::fft::a_weighting(Float(3135.96)), Float(1.20425069)));    // G7
    REQUIRE(approx_equal(neo::fft::a_weighting(Float(6271.93)), Float(-0.09973374)));   // G8
    REQUIRE(approx_equal(neo::fft::a_weighting(Float(12543.85)), Float(-4.28495301)));  // G9
    return true;
}

TEST_CASE("a_weighting")
{
    REQUIRE(test_a_weighting<float>());
    REQUIRE(test_a_weighting<double>());
}
