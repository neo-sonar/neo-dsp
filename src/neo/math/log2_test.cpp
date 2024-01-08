// SPDX-License-Identifier: MIT

#include "log2.hpp"

#include <neo/complex.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

namespace {
template<typename T>
inline constexpr auto has_log2 = requires(T const& t) { neo::math::log2(t); };
}  // namespace

TEMPLATE_TEST_CASE("neo/math: log2", "", short, int, unsigned, long, float, double)
{
    using Float = TestType;

    STATIC_REQUIRE(has_log2<Float>);

    REQUIRE(neo::math::log2(Float{1}) == Catch::Approx(0.0));
    REQUIRE(neo::math::log2(Float{2}) == Catch::Approx(1.0));
    REQUIRE(neo::math::log2(Float{16}) == Catch::Approx(4.0));
}
