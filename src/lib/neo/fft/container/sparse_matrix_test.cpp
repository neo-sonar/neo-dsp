#include "sparse_matrix.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/fft/container: sparse_matrix", "", float, double)
{
    using Float = TestType;

    auto isZero = [](auto x) { return std::equal_to{}(x, Float(0)); };

    auto lhs = KokkosEx::mdarray<Float, Kokkos::dextents<std::size_t, 2>>{16, 32};
    std::fill(lhs.data(), std::next(lhs.data(), std::ssize(lhs)), Float(1));

    auto rhs = neo::sparse_matrix<Float>{16, 32};
    REQUIRE(rhs.rows() == 16);
    REQUIRE(rhs.columns() == 32);

    auto accumulator = std::vector<Float>(lhs.extent(1));
    neo::multiply_elementwise_accumulate_columnwise<Float>(lhs.to_mdspan(), rhs, std::span<Float>{accumulator});
    REQUIRE(std::all_of(accumulator.begin(), accumulator.end(), isZero));

    rhs.insert(0, 0, Float(2));
    neo::multiply_elementwise_accumulate_columnwise<Float>(lhs.to_mdspan(), rhs, std::span<Float>{accumulator});
    REQUIRE(accumulator[0] == Catch::Approx(Float(2)));
    REQUIRE(std::all_of(std::next(accumulator.begin()), accumulator.end(), isZero));
    // std::fill(accumulator.begin(), accumulator.end(), Float(0));

    auto other = neo::sparse_matrix<Float>{lhs.to_mdspan(), [](auto, auto, auto v) { return v >= Float(1); }};
    REQUIRE(other.rows() == lhs.extent(0));
    REQUIRE(other.columns() == lhs.extent(1));
    REQUIRE(other.value_container().size() == lhs.size());

    other = neo::sparse_matrix<Float>{lhs.to_mdspan(), [](auto, auto, auto v) { return v >= Float(2); }};
    REQUIRE(other.rows() == lhs.extent(0));
    REQUIRE(other.columns() == lhs.extent(1));
    REQUIRE(other.value_container().size() == 0);

    auto row = std::vector<Float>(lhs.extent(1));
    std::fill(row.begin(), row.end(), Float(2));
    other.insert_row(0, std::span{row}, [](auto, auto, auto) { return true; });
    REQUIRE(other.value_container().size() == 32);
    REQUIRE(other.column_container().size() == 32);
}
