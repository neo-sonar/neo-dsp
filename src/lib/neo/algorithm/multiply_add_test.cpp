#include "multiply_add.hpp"

#include <neo/algorithm/allmatch.hpp>
#include <neo/math/float_equality.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

template<typename Float>
auto test()
{
    auto isZero = [](auto x) { return neo::float_equality::exact(x, Float(0)); };

    auto lhs = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{16, 32};
    std::fill(lhs.data(), std::next(lhs.data(), std::ssize(lhs)), Float(1));

    auto rhs = neo::sparse_matrix<Float>{16, 32};
    REQUIRE(rhs.rows() == 16);
    REQUIRE(rhs.columns() == 32);

    auto accumulator = std::vector<Float>(lhs.extent(1));
    auto acc         = stdex::mdspan{accumulator.data(), stdex::extents{accumulator.size()}};
    auto left_row0   = stdex::submdspan(lhs.to_mdspan(), 0, stdex::full_extent);

    neo::multiply_add(left_row0, rhs, 0, acc, acc);
    REQUIRE(neo::allmatch(acc, isZero));

    rhs.insert(0, 0, Float(2));
    neo::multiply_add(left_row0, rhs, 0, acc, acc);
    REQUIRE(accumulator[0] == Catch::Approx(Float(2)));
    REQUIRE(neo::allmatch(stdex::submdspan(acc, std::tuple{1, acc.extent(0)}), isZero));

    rhs.insert(0, 1, Float(4));
    neo::fill(acc, Float(0));
    neo::multiply_add(left_row0, rhs, 0, acc, acc);
    REQUIRE(accumulator[0] == Catch::Approx(Float(2)));
    REQUIRE(accumulator[1] == Catch::Approx(Float(4)));
    REQUIRE(neo::allmatch(stdex::submdspan(acc, std::tuple{2, acc.extent(0)}), isZero));
}

TEMPLATE_TEST_CASE("neo/algorithm: multiply_add(sparse_matrix)", "", float, double) { test<TestType>(); }

#if defined(NEO_HAS_BUILTIN_FLOAT16)
TEMPLATE_TEST_CASE("neo/algorithm: multiply_add(sparse_matrix)", "", _Float16) { test<TestType>(); }
#endif
