#include "simd.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>

#if defined(NEO_HAS_SSE2)

TEST_CASE("neo/math: simd::float32x4_t")
{
    auto output = std::array<float, 4>{};
    auto reg    = neo::simd::float32x4_t::broadcast(1.0F);
    reg.store_unaligned(output.data());
    for (auto val : output) {
        REQUIRE(val == Catch::Approx(1.0));
    }
}

TEST_CASE("neo/math: simd::float64x2_t")
{
    auto output = std::array<double, 2>{};
    auto reg    = neo::simd::float64x2_t::broadcast(1.0F);
    reg.store_unaligned(output.data());
    for (auto val : output) {
        REQUIRE(val == Catch::Approx(1.0));
    }
}

#endif
