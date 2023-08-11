#include "copy.hpp"

#include <neo/fft/algorithm/allclose.hpp>
#include <neo/fft/algorithm/fill.hpp>
#include <neo/fft/testing/testing.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEMPLATE_TEST_CASE("neo/fft/algorithm: copy(in_vector)", "", float, double, std::complex<float>, std::complex<double>)
{
    using Float     = decltype(neo::fft::float_or_complex_value_type<TestType>());
    auto const size = GENERATE(as<std::size_t>{}, 1, 2, 33, 128);

    SECTION("vector")
    {
        auto const makeVector = [size](Float val) {
            auto vec = KokkosEx::mdarray<Float, Kokkos::dextents<std::size_t, 1>>{size};
            neo::fft::fill(vec.to_mdspan(), val);
            return vec;
        };

        auto out      = makeVector(Float(0));
        auto const in = makeVector(Float(1));
        neo::fft::copy(in.to_mdspan(), out.to_mdspan());
        REQUIRE(neo::fft::allclose(in.to_mdspan(), out.to_mdspan()));
    }

    SECTION("matrix")
    {
        auto const makeMatrix = [size](Float val) {
            auto vec = KokkosEx::mdarray<Float, Kokkos::dextents<std::size_t, 2>>{size, size};
            neo::fft::fill(vec.to_mdspan(), val);
            return vec;
        };

        auto out      = makeMatrix(Float(0));
        auto const in = makeMatrix(Float(1));
        neo::fft::copy(in.to_mdspan(), out.to_mdspan());
        REQUIRE(neo::fft::allclose(in.to_mdspan(), out.to_mdspan()));
    }
}
