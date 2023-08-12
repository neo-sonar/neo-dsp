#include "power.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE(
    "neo/math: power",
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

    REQUIRE(neo::power<Int(1)>(Int(0)) == Int(1));
    REQUIRE(neo::power<Int(1)>(Int(1)) == Int(1));
    REQUIRE(neo::power<Int(1)>(Int(2)) == Int(1));
    REQUIRE(neo::power<Int(1)>(Int(3)) == Int(1));

    REQUIRE(neo::power<Int(2)>(Int(0)) == Int(1));
    REQUIRE(neo::power<Int(2)>(Int(1)) == Int(2));
    REQUIRE(neo::power<Int(2)>(Int(2)) == Int(4));
    REQUIRE(neo::power<Int(2)>(Int(3)) == Int(8));

    REQUIRE(neo::power<Int(3)>(Int(0)) == Int(1));
    REQUIRE(neo::power<Int(3)>(Int(1)) == Int(3));
    REQUIRE(neo::power<Int(3)>(Int(2)) == Int(9));
    REQUIRE(neo::power<Int(3)>(Int(3)) == Int(27));

    REQUIRE(neo::power<Int(10)>(Int(0)) == Int(1));
    REQUIRE(neo::power<Int(10)>(Int(1)) == Int(10));
    REQUIRE(neo::power<Int(10)>(Int(2)) == Int(100));
}
