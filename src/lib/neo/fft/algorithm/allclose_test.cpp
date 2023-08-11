#include "allclose.hpp"

#include <neo/fft/algorithm/fill.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEMPLATE_TEST_CASE("neo/fft/algorithm: allclose(in_vector)", "", float, double, std::complex<float>, std::complex<double>)
{
    using Float = TestType;

    auto const size       = GENERATE(as<std::size_t>{}, 1, 2, 33, 128);
    auto const makeVector = [size](Float val) {
        auto vec = KokkosEx::mdarray<Float, Kokkos::dextents<std::size_t, 1>>{size};
        neo::fft::fill(vec.to_mdspan(), val);
        return vec;
    };

    auto const zeros = makeVector(Float(0));
    auto const ones  = makeVector(Float(1));

    REQUIRE(neo::fft::allclose(zeros.to_mdspan(), zeros.to_mdspan()));
    REQUIRE(neo::fft::allclose(ones.to_mdspan(), ones.to_mdspan()));

    REQUIRE_FALSE(neo::fft::allclose(zeros.to_mdspan(), ones.to_mdspan()));
    REQUIRE_FALSE(neo::fft::allclose(ones.to_mdspan(), zeros.to_mdspan()));
}
