#include "multiply_sum_columns.hpp"

#include <neo/math/float_equality.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/algorithm: multiply_sum_columns(sparse_matrix)", "", float, double, long double)
{
    using Float = TestType;

    auto isZero = [](auto x) { return neo::float_equality::exact(x, Float(0)); };

    auto lhs = KokkosEx::mdarray<Float, Kokkos::dextents<std::size_t, 2>>{16, 32};
    std::fill(lhs.data(), std::next(lhs.data(), std::ssize(lhs)), Float(1));

    auto rhs = neo::sparse_matrix<Float>{16, 32};
    REQUIRE(rhs.rows() == 16);
    REQUIRE(rhs.columns() == 32);

    auto accumulator = std::vector<Float>(lhs.extent(1));
    auto acc         = Kokkos::mdspan{accumulator.data(), Kokkos::extents{accumulator.size()}};

    neo::multiply_sum_columns(lhs.to_mdspan(), rhs, acc);
    REQUIRE(std::all_of(accumulator.begin(), accumulator.end(), isZero));

    rhs.insert(0, 0, Float(2));
    neo::multiply_sum_columns(lhs.to_mdspan(), rhs, acc);
    REQUIRE(accumulator[0] == Catch::Approx(Float(2)));
    REQUIRE(std::all_of(std::next(accumulator.begin()), accumulator.end(), isZero));
}
