// SPDX-License-Identifier: MIT

#include "uniform_partition.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/convolution: uniform_partition", "", float, double)
{
    using Float = TestType;

    SECTION("mono")
    {
        auto const impulse    = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{1, 4096};
        auto const partitions = neo::uniform_partition(impulse.to_mdspan(), 128);
        REQUIRE(partitions.extent(0) == 1);
        REQUIRE(partitions.extent(1) == 32);
        REQUIRE(partitions.extent(2) == 129);
    }

    SECTION("stereo")
    {
        auto const impulse    = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{2, 4096};
        auto const partitions = neo::uniform_partition(impulse.to_mdspan(), 128);
        REQUIRE(partitions.extent(0) == 2);
        REQUIRE(partitions.extent(1) == 32);
        REQUIRE(partitions.extent(2) == 129);
    }

    SECTION("stereo - odd")
    {
        auto const impulse    = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{2, 4095};
        auto const partitions = neo::uniform_partition(impulse.to_mdspan(), 128);
        REQUIRE(partitions.extent(0) == 2);
        REQUIRE(partitions.extent(1) == 32);
        REQUIRE(partitions.extent(2) == 129);
    }
}
