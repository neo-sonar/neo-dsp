// SPDX-License-Identifier: MIT

#include "radix4_plan.hpp"

#include <neo/algorithm/allclose.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/complex/scalar_complex.hpp>
#include <neo/fft.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEMPLATE_TEST_CASE(
    "neo/fft: experimental::radix4_plan",
    "",
    std::complex<float>,
    std::complex<double>,
    std::complex<long double>,
    neo::complex64,
    neo::complex128
)
{
    using Complex = TestType;
    using Float   = typename Complex::value_type;

    auto const order = GENERATE(as<neo::fft::order>{}, 1, 2, 3, 4, 5, 6, 7, 8, 9);
    CAPTURE(order);

    SECTION("inplace DIT")
    {
        auto plan        = neo::fft::experimental::radix4_plan<Complex, true>{order};
        auto const noise = neo::generate_noise_signal<Complex>(plan.size(), Catch::getSeed());

        auto copy = noise;
        auto io   = copy.to_mdspan();

        neo::fft::fft(plan, io);
        neo::fft::ifft(plan, io);

        neo::scale(Float(1) / static_cast<Float>(plan.size()), io);
        REQUIRE(neo::allclose(noise.to_mdspan(), io, Float(0.001)));
    }

    SECTION("inplace DIF")
    {
        auto plan        = neo::fft::experimental::radix4_plan<Complex, false>{order};
        auto const noise = neo::generate_noise_signal<Complex>(plan.size(), Catch::getSeed());

        auto copy = noise;
        auto io   = copy.to_mdspan();

        neo::fft::fft(plan, io);
        neo::fft::ifft(plan, io);

        neo::scale(Float(1) / static_cast<Float>(plan.size()), io);
        REQUIRE(neo::allclose(noise.to_mdspan(), io, Float(0.001)));
    }
}
