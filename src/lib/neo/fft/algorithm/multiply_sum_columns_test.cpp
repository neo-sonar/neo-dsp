#include "multiply_sum_columns.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/fft/algorithm: multiply_sum_columns(sparse_matrix)", "", float, double)
{
    using Float = TestType;

    auto isZero = [](auto x) { return std::equal_to{}(x, Float(0)); };

    auto lhs = KokkosEx::mdarray<Float, Kokkos::dextents<std::size_t, 2>>{16, 32};
    std::fill(lhs.data(), std::next(lhs.data(), std::ssize(lhs)), Float(1));

    auto rhs = neo::fft::sparse_matrix<Float>{16, 32};
    REQUIRE(rhs.rows() == 16);
    REQUIRE(rhs.columns() == 32);

    auto accumulator = std::vector<Float>(lhs.extent(1));
    neo::fft::multiply_sum_columns<Float>(lhs.to_mdspan(), rhs, std::span<Float>{accumulator});
    REQUIRE(std::all_of(accumulator.begin(), accumulator.end(), isZero));

    rhs.insert(0, 0, Float(2));
    neo::fft::multiply_sum_columns<Float>(lhs.to_mdspan(), rhs, std::span<Float>{accumulator});
    REQUIRE(accumulator[0] == Catch::Approx(Float(2)));
    REQUIRE(std::all_of(std::next(accumulator.begin()), accumulator.end(), isZero));
}
