#include "neo/fft/convolution/container/sparse_matrix.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("sparse_matrix")
{
    auto lhs = KokkosEx::mdarray<float, Kokkos::dextents<std::size_t, 2>>{16, 32};
    std::fill(lhs.data(), std::next(lhs.data(), std::ssize(lhs)), 1.0F);

    auto rhs = neo::sparse_matrix<float>{16, 32};
    REQUIRE(rhs.rows() == 16);
    REQUIRE(rhs.columns() == 32);

    auto accumulator = std::vector<float>(lhs.extent(1));
    neo::multiply_elementwise_accumulate_columnwise<float>(lhs.to_mdspan(), rhs, std::span<float>{accumulator});
    REQUIRE(std::all_of(accumulator.begin(), accumulator.end(), [](auto x) { return x == 0.0F; }));

    rhs.insert(0, 0, 2.0F);
    neo::multiply_elementwise_accumulate_columnwise<float>(lhs.to_mdspan(), rhs, std::span<float>{accumulator});
    REQUIRE(accumulator[0] == Catch::Approx(2.0F));
    REQUIRE(std::all_of(std::next(accumulator.begin()), accumulator.end(), [](auto x) { return x == 0.0F; }));
    // std::fill(accumulator.begin(), accumulator.end(), 0.0F);

    auto other = neo::sparse_matrix<float>{lhs.to_mdspan(), [](auto, auto, auto v) { return v >= 1.0F; }};
    REQUIRE(other.rows() == lhs.extent(0));
    REQUIRE(other.columns() == lhs.extent(1));
    REQUIRE(other.value_container().size() == lhs.size());

    other = neo::sparse_matrix<float>{lhs.to_mdspan(), [](auto, auto, auto v) { return v >= 2.0F; }};
    REQUIRE(other.rows() == lhs.extent(0));
    REQUIRE(other.columns() == lhs.extent(1));
    REQUIRE(other.value_container().size() == 0);

    auto row = std::vector<float>(lhs.extent(1));
    std::fill(row.begin(), row.end(), 2.0F);
    other.insert_row(0, std::span{row}, [](auto, auto, auto) { return true; });
    REQUIRE(other.value_container().size() == 32);
    REQUIRE(other.column_container().size() == 32);
}
