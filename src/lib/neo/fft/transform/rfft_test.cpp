#include "rfft.hpp"

#include <neo/fft/algorithm/allclose.hpp>
#include <neo/fft/transform/radix2.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <array>
#include <complex>
#include <random>

TEMPLATE_TEST_CASE("neo/fft/transform/rfft: extract_two_real_dfts", "", float, double)
{
    using Float = TestType;

    static constexpr auto n     = 8UL;
    static constexpr auto order = neo::fft::ilog2(n);

    auto rng    = std::mt19937{std::random_device{}()};
    auto dist   = std::uniform_real_distribution<Float>{-1, 1};
    auto random = [&dist, &rng] { return dist(rng); };

    auto a = std::array<Float, n>{};
    auto b = std::array<Float, n>{};
    std::generate(a.begin(), a.end(), random);
    std::generate(b.begin(), b.end(), random);

    auto fft  = neo::fft::c2c_radix2_plan<std::complex<Float>>{n};
    auto rfft = neo::fft::rfft_radix2_plan<Float>{order};

    auto a_rev = std::array<std::complex<Float>, n / 2 + 1>{};
    auto b_rev = std::array<std::complex<Float>, n / 2 + 1>{};
    rfft(a, a_rev);
    rfft(b, b_rev);

    auto inout = std::array<std::complex<Float>, n>{};
    std::transform(a.begin(), a.end(), b.begin(), inout.begin(), [](auto ra, auto rb) { return std::complex{ra, rb}; });

    fft(Kokkos::mdspan{inout.data(), Kokkos::extents{inout.size()}}, neo::fft::direction::forward);

    auto ca = std::array<std::complex<Float>, n / 2 + 1>{};
    auto cb = std::array<std::complex<Float>, n / 2 + 1>{};
    neo::fft::extract_two_real_dfts<Float>(inout, ca, cb);

    CHECK(neo::fft::allclose(a_rev, ca));
    CHECK(neo::fft::allclose(b_rev, cb));
}
