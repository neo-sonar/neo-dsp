// SPDX-License-Identifier: MIT
#include "bit_ceil.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/math: bit_ceil", "", std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t)
{
    using UInt = TestType;

    REQUIRE(neo::bit_ceil(UInt(1)) == UInt(1));
    REQUIRE(neo::bit_ceil(UInt(2)) == UInt(2));
    REQUIRE(neo::bit_ceil(UInt(3)) == UInt(4));
    REQUIRE(neo::bit_ceil(UInt(4)) == UInt(4));
    REQUIRE(neo::bit_ceil(UInt(100)) == UInt(128));
}

TEMPLATE_TEST_CASE("neo/math: detail::bit_ceil_fallback", "", std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t)
{
    using UInt = TestType;

    REQUIRE(neo::detail::bit_ceil_fallback(UInt(1)) == UInt(1));
    REQUIRE(neo::detail::bit_ceil_fallback(UInt(2)) == UInt(2));
    REQUIRE(neo::detail::bit_ceil_fallback(UInt(3)) == UInt(4));
    REQUIRE(neo::detail::bit_ceil_fallback(UInt(4)) == UInt(4));
    REQUIRE(neo::detail::bit_ceil_fallback(UInt(100)) == UInt(128));
}
