#include "multiply_accumulate.hpp"

#include <neo/algorithm/all_of.hpp>
#include <neo/math/float_equality.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/fft/convolution: multiply_accumulate(sparse_matrix)", "", float, double, long double)
{
    using Float = TestType;

    auto isZero = [](auto x) { return neo::float_equality::exact(x, Float(0)); };

    auto lhs = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{16, 32};
    std::fill(lhs.data(), std::next(lhs.data(), std::ssize(lhs)), Float(1));

    auto rhs = neo::sparse_matrix<Float>{16, 32};
    REQUIRE(rhs.rows() == 16);
    REQUIRE(rhs.columns() == 32);

    auto accumulator = std::vector<Float>(lhs.extent(1));
    auto acc         = stdex::mdspan{accumulator.data(), stdex::extents{accumulator.size()}};
    auto left_row0   = stdex::submdspan(lhs.to_mdspan(), 0, stdex::full_extent);

    neo::fft::multiply_accumulate(left_row0, rhs, 0, acc);
    REQUIRE(neo::all_of(acc, isZero));

    rhs.insert(0, 0, Float(2));
    neo::fft::multiply_accumulate(left_row0, rhs, 0, acc);
    REQUIRE(accumulator[0] == Catch::Approx(Float(2)));
    REQUIRE(neo::all_of(stdex::submdspan(acc, std::tuple{1, acc.extent(0)}), isZero));
}
