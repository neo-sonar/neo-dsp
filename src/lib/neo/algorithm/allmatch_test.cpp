#include "allmatch.hpp"

#include <neo/algorithm/fill.hpp>
#include <neo/complex/scalar_complex.hpp>
#include <neo/math/float_equality.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEMPLATE_TEST_CASE(
    "neo/algorithm: allmatch",
    "",
    float,
    double,
    std::complex<float>,
    std::complex<double>,
    neo::complex64,
    neo::complex128
)
{
    using Float = neo::real_or_complex_value_t<TestType>;

    auto const size  = GENERATE(as<std::size_t>{}, 2, 33, 128);
    auto const exact = [](auto lhs, auto rhs) { return neo::float_equality::exact(lhs, rhs); };

    SECTION("vector")
    {
        auto const makeVector = [size](Float val) {
            auto vec = stdex::mdarray<Float, stdex::dextents<std::size_t, 1>>{size};
            neo::fill(vec.to_mdspan(), val);
            return vec;
        };

        auto const zeros      = makeVector(Float(0));
        auto const ones       = makeVector(Float(1));
        auto const almostOnes = makeVector(Float(0.98));

        REQUIRE_FALSE(neo::allmatch(ones.to_mdspan(), zeros.to_mdspan(), exact));
        REQUIRE_FALSE(neo::allmatch(ones.to_mdspan(), almostOnes.to_mdspan(), exact));

        REQUIRE(neo::allmatch(zeros.to_mdspan(), zeros.to_mdspan(), exact));
        REQUIRE(neo::allmatch(ones.to_mdspan(), ones.to_mdspan(), exact));

        REQUIRE(neo::allmatch(zeros.to_mdspan(), [exact](auto val) { return exact(val, Float(0)); }));
        REQUIRE(neo::allmatch(ones.to_mdspan(), [exact](auto val) { return exact(val, Float(1)); }));
    }

    SECTION("matrix")
    {
        auto const makeMatrix = [size](Float val) {
            auto vec = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{size, size * 2UL};
            neo::fill(vec.to_mdspan(), val);
            return vec;
        };

        auto const zeros      = makeMatrix(Float(0));
        auto const ones       = makeMatrix(Float(1));
        auto const almostOnes = makeMatrix(Float(0.98));

        REQUIRE_FALSE(neo::allmatch(ones.to_mdspan(), zeros.to_mdspan(), exact));
        REQUIRE_FALSE(neo::allmatch(ones.to_mdspan(), almostOnes.to_mdspan(), exact));

        REQUIRE(neo::allmatch(zeros.to_mdspan(), zeros.to_mdspan(), exact));
        REQUIRE(neo::allmatch(ones.to_mdspan(), ones.to_mdspan(), exact));

        REQUIRE(neo::allmatch(zeros.to_mdspan(), [exact](auto val) { return exact(val, Float(0)); }));
        REQUIRE(neo::allmatch(ones.to_mdspan(), [exact](auto val) { return exact(val, Float(1)); }));
    }
}
