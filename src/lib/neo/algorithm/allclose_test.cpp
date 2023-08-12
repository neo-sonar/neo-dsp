#include "allclose.hpp"

#include <neo/algorithm/fill.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEMPLATE_TEST_CASE("neo/algorithm: allclose", "", float, double, long double, std::complex<float>, std::complex<double>, std::complex<long double>)
{
    using Float = neo::real_or_complex_value_t<TestType>;

    auto const size = GENERATE(as<std::size_t>{}, 2, 33, 128);

    SECTION("vector")
    {
        auto const makeVector = [size](Float val) {
            auto vec = KokkosEx::mdarray<Float, Kokkos::dextents<std::size_t, 1>>{size};
            neo::fill(vec.to_mdspan(), val);
            return vec;
        };

        auto const zeros      = makeVector(Float(0));
        auto const ones       = makeVector(Float(1));
        auto const almostOnes = makeVector(Float(0.95));

        REQUIRE_FALSE(neo::allclose(ones.to_mdspan(), zeros.to_mdspan()));
        REQUIRE_FALSE(neo::allclose(ones.to_mdspan(), almostOnes.to_mdspan()));

        REQUIRE(neo::allclose(zeros.to_mdspan(), zeros.to_mdspan()));
        REQUIRE(neo::allclose(ones.to_mdspan(), ones.to_mdspan()));
        REQUIRE(neo::allclose(ones.to_mdspan(), almostOnes.to_mdspan(), Float(0.7)));

        if constexpr (neo::current_contracts_check_mode == neo::contracts_check_mode::exception) {
            auto sub = KokkosEx::submdspan(ones.to_mdspan(), std::tuple{0, size - 1UL});
            REQUIRE_THROWS(neo::allclose(sub, zeros.to_mdspan()));
        }
    }

    SECTION("matrix")
    {
        auto const makeMatrix = [size](Float val) {
            auto vec = KokkosEx::mdarray<Float, Kokkos::dextents<std::size_t, 2>>{size, size * 2UL};
            neo::fill(vec.to_mdspan(), val);
            return vec;
        };

        auto const zeros      = makeMatrix(Float(0));
        auto const ones       = makeMatrix(Float(1));
        auto const almostOnes = makeMatrix(Float(0.95));

        REQUIRE_FALSE(neo::allclose(ones.to_mdspan(), zeros.to_mdspan()));
        REQUIRE_FALSE(neo::allclose(ones.to_mdspan(), almostOnes.to_mdspan()));

        REQUIRE(neo::allclose(zeros.to_mdspan(), zeros.to_mdspan()));
        REQUIRE(neo::allclose(ones.to_mdspan(), ones.to_mdspan()));
        REQUIRE(neo::allclose(ones.to_mdspan(), almostOnes.to_mdspan(), Float(0.7)));

        if constexpr (neo::current_contracts_check_mode == neo::contracts_check_mode::exception) {
            auto sub = KokkosEx::submdspan(ones.to_mdspan(), std::tuple{0, size - 1UL}, Kokkos::full_extent);
            REQUIRE_THROWS(neo::allclose(sub, zeros.to_mdspan()));
        }
    }
}
