#include "allclose.hpp"

#include <neo/fft/algorithm/fill.hpp>
#include <neo/fft/testing/testing.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEMPLATE_TEST_CASE("neo/fft/algorithm: allclose(in_vector)", "", float, double, std::complex<float>, std::complex<double>)
{
    using Float = neo::fft::float_or_complex_value_type_t<TestType>;

    auto const size       = GENERATE(as<std::size_t>{}, 2, 33, 128);
    auto const makeVector = [size](Float val) {
        auto vec = KokkosEx::mdarray<Float, Kokkos::dextents<std::size_t, 1>>{size};
        neo::fft::fill(vec.to_mdspan(), val);
        return vec;
    };

    auto const zeros      = makeVector(Float(0));
    auto const ones       = makeVector(Float(1));
    auto const almostOnes = makeVector(Float(0.95));

    REQUIRE_FALSE(neo::fft::allclose(ones.to_mdspan(), zeros.to_mdspan()));
    REQUIRE_FALSE(neo::fft::allclose(ones.to_mdspan(), almostOnes.to_mdspan()));

    REQUIRE(neo::fft::allclose(zeros.to_mdspan(), zeros.to_mdspan()));
    REQUIRE(neo::fft::allclose(ones.to_mdspan(), ones.to_mdspan()));
    REQUIRE(neo::fft::allclose(ones.to_mdspan(), almostOnes.to_mdspan(), Float(0.7)));

    if constexpr (neo::fft::current_contracts_check_mode == neo::fft::contracts_check_mode::exception) {
        auto sub = KokkosEx::submdspan(ones.to_mdspan(), std::tuple{0, size - 1UL});
        REQUIRE_THROWS(neo::fft::allclose(sub, zeros.to_mdspan()));
    }
}
