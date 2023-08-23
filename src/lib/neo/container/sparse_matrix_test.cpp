#include "sparse_matrix.hpp"

#include <neo/algorithm/fill.hpp>
#include <neo/math/float_equality.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/container: sparse_matrix", "", float, double, long double, std::complex<float>, std::complex<double>, std::complex<long double>)
{
    using Scalar = TestType;
    using Float  = neo::real_or_complex_value_t<Scalar>;

    auto greaterEqualToX = [](auto x) {
        return [x](auto, auto, auto v) {
            if constexpr (neo::complex<Scalar>) {
                return v.real() >= Float(x);
            } else {
                return v >= Float(x);
            }
        };
    };

    auto greaterEqualOne = greaterEqualToX(Float(1));
    auto greaterEqualTwo = greaterEqualToX(Float(2));

    SECTION("insert")
    {
        auto matrix = neo::sparse_matrix<Float>{16, 32};
        REQUIRE(matrix(0, 0) == Catch::Approx(Float(0)));
        matrix.insert(0, 0, Float(2));
        REQUIRE(matrix(0, 0) == Catch::Approx(Float(2)));

        matrix.insert(0, 1, Float(4));
        REQUIRE(matrix(0, 0) == Catch::Approx(Float(2)));
        REQUIRE(matrix(0, 1) == Catch::Approx(Float(4)));

        REQUIRE(matrix(0, 4) == Catch::Approx(Float(0)));
        matrix.insert(0, 4, Float(42));
        REQUIRE(matrix(0, 4) == Catch::Approx(Float(42)));
    }

    auto dense = stdex::mdarray<Scalar, stdex::dextents<std::size_t, 2>>{16, 32};
    neo::fill(dense.to_mdspan(), Scalar(Float(1)));

    auto sparse = neo::sparse_matrix<Scalar>{dense.to_mdspan(), greaterEqualOne};
    REQUIRE(sparse.rows() == dense.extent(0));
    REQUIRE(sparse.columns() == dense.extent(1));
    REQUIRE(sparse.extent(0) == dense.extent(0));
    REQUIRE(sparse.extent(1) == dense.extent(1));
    REQUIRE(sparse.size() == dense.size());
    REQUIRE(sparse.extents() == dense.extents());
    REQUIRE(sparse.value_container().size() == dense.size());

    sparse = neo::sparse_matrix<Scalar>{dense.to_mdspan(), greaterEqualTwo};
    REQUIRE(sparse.rows() == dense.extent(0));
    REQUIRE(sparse.columns() == dense.extent(1));
    REQUIRE(sparse.size() == dense.size());
    REQUIRE(sparse.value_container().size() == 0);
}
