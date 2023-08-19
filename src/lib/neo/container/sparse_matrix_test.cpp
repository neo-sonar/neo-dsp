#include "sparse_matrix.hpp"

#include <neo/algorithm/fill.hpp>
#include <neo/math/float_equality.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/container: sparse_matrix", "", float, double, (long double), std::complex<float>, std::complex<double>, std::complex<long double>)
{
    using Scalar = TestType;
    using Float  = neo::real_or_complex_value_t<Scalar>;

    SECTION("insert")
    {
        auto rhs = neo::sparse_matrix<Float>{16, 32};
        rhs.insert(0, 0, Float(2));
        REQUIRE(rhs(0, 0) == Catch::Approx(Float(2)));

        rhs.insert(0, 1, Float(4));
        REQUIRE(rhs(0, 0) == Catch::Approx(Float(2)));
        REQUIRE(rhs(0, 1) == Catch::Approx(Float(4)));
    }

    auto greaterEqualOne = [](auto, auto, auto v) { return std::real(v) >= Float(1); };
    auto greaterEqualTwo = [](auto, auto, auto v) { return std::real(v) >= Float(2); };

    auto lhs = stdex::mdarray<Scalar, stdex::dextents<std::size_t, 2>>{16, 32};
    neo::fill(lhs.to_mdspan(), Scalar(Float(1)));

    auto other = neo::sparse_matrix<Scalar>{lhs.to_mdspan(), greaterEqualOne};
    REQUIRE(other.rows() == lhs.extent(0));
    REQUIRE(other.columns() == lhs.extent(1));
    REQUIRE(other.size() == lhs.size());
    REQUIRE(other.value_container().size() == lhs.size());

    other = neo::sparse_matrix<Scalar>{lhs.to_mdspan(), greaterEqualTwo};
    REQUIRE(other.rows() == lhs.extent(0));
    REQUIRE(other.columns() == lhs.extent(1));
    REQUIRE(other.size() == lhs.size());
    REQUIRE(other.value_container().size() == 0);

    auto rowData = std::vector<Scalar>(lhs.extent(1));
    auto row     = stdex::mdspan{rowData.data(), stdex::extents{rowData.size()}};
    neo::fill(row, Scalar(Float(2)));

    other.insert_row(0, row, [](auto, auto, auto) { return true; });
    REQUIRE(other.value_container().size() == 32);
    REQUIRE(other.column_container().size() == 32);

    auto const half = lhs.extent(1) / 2;
    other.insert_row(1, row, [half](auto, auto col, auto) { return col < half; });
    REQUIRE(neo::float_equality::exact(other(1, half), Scalar(Float(0))));
    REQUIRE(neo::float_equality::exact(other(1, half - 1), Scalar(Float(2))));

    other.insert_row(1, row, [](auto, auto, auto) { return true; });
    REQUIRE(other.value_container().size() == 64);
    REQUIRE(other.column_container().size() == 64);

    other.insert_row(1, row, [](auto, auto col, auto) { return col % 2U == 0; });
    REQUIRE(other.value_container().size() == 48);
    REQUIRE(other.column_container().size() == 48);
}
