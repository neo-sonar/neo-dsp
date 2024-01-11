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

#include <random>

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

    auto const order = GENERATE(as<neo::fft::order>{}, 1, 2, 3, 4, 5, 6, 7, 8);
    auto const size  = neo::ipow<size_t(4)>(size_t(order));
    CAPTURE(order);

    auto dirac = stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>>{size};
    dirac(0)   = Complex{Float(1)};

    SECTION("DIT")
    {
        auto dit = neo::fft::experimental::radix4_plan<Complex, true>{order};

        auto copy = dirac;
        auto io   = copy.to_mdspan();
        neo::fft::fft(dit, io);

        for (auto i{0}; i < int(dit.size()); ++i) {
            CAPTURE(i);
            REQUIRE(io[i].real() == Catch::Approx(Float(1)));
            REQUIRE(io[i].imag() == Catch::Approx(Float(0)));
        }
    }

    SECTION("DIF")
    {
        auto dif = neo::fft::experimental::radix4_plan<Complex, false>{order};

        auto copy = dirac;
        auto io   = copy.to_mdspan();
        neo::fft::fft(dif, io);

        for (auto i{0}; i < int(dif.size()); ++i) {
            CAPTURE(i);
            REQUIRE(io[i].real() == Catch::Approx(Float(1)));
            REQUIRE(io[i].imag() == Catch::Approx(Float(0)));
        }
    }
}
