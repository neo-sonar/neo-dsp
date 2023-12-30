// SPDX-License-Identifier: MIT

#include "ilog2.hpp"

#include <catch2/catch_template_test_macros.hpp>

#include <cstdint>

TEMPLATE_TEST_CASE(
    "neo/math: ilog2",
    "",
    std::int8_t,
    std::uint8_t,
    std::int16_t,
    std::uint16_t,
    std::int32_t,
    std::uint32_t,
    std::int64_t,
    std::uint64_t
)
{
    using Int = TestType;

    STATIC_REQUIRE(std::same_as<decltype(neo::ilog2(std::declval<Int>())), Int>);

    REQUIRE(neo::ilog2(Int(1)) == Int(0));
    REQUIRE(neo::ilog2(Int(2)) == Int(1));
    REQUIRE(neo::ilog2(Int(4)) == Int(2));
    REQUIRE(neo::ilog2(Int(8)) == Int(3));
    REQUIRE(neo::ilog2(Int(16)) == Int(4));
    REQUIRE(neo::ilog2(Int(32)) == Int(5));
    REQUIRE(neo::ilog2(Int(64)) == Int(6));

    if constexpr (sizeof(Int) > 1) {
        REQUIRE(neo::ilog2(Int(128)) == Int(7));
        REQUIRE(neo::ilog2(Int(256)) == Int(8));
        REQUIRE(neo::ilog2(Int(512)) == Int(9));
        REQUIRE(neo::ilog2(Int(1024)) == Int(10));
        REQUIRE(neo::ilog2(Int(2048)) == Int(11));
        REQUIRE(neo::ilog2(Int(4096)) == Int(12));
        REQUIRE(neo::ilog2(Int(8192)) == Int(13));
    }
}
