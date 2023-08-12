#include "next_power_of_two.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/math: next_power_of_two", "", std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t)
{
    using UInt = TestType;

    REQUIRE(neo::fft::next_power_of_two(UInt(1)) == UInt(1));
    REQUIRE(neo::fft::next_power_of_two(UInt(2)) == UInt(2));
    REQUIRE(neo::fft::next_power_of_two(UInt(3)) == UInt(4));
    REQUIRE(neo::fft::next_power_of_two(UInt(4)) == UInt(4));
    REQUIRE(neo::fft::next_power_of_two(UInt(100)) == UInt(128));
}

TEMPLATE_TEST_CASE(
    "neo/math: detail::next_power_of_two_fallback",
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
