#include "rfft.hpp"

#include <neo/fft/algorithm/allclose.hpp>
#include <neo/fft/testing/testing.hpp>
#include <neo/fft/transform/radix2.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <array>
#include <complex>
#include <random>

TEMPLATE_TEST_CASE("neo/fft/transform/rfft: test_data(rfft_plan)", "", float, double)
{
    using Float = TestType;

    auto paths = GENERATE(
        neo::fft::test_path{"./test_data/r2c_8_input.csv", "./test_data/r2c_8_output.csv"},
        neo::fft::test_path{"./test_data/r2c_16_input.csv", "./test_data/r2c_16_output.csv"},
        neo::fft::test_path{"./test_data/r2c_32_input.csv", "./test_data/r2c_32_output.csv"},
        neo::fft::test_path{"./test_data/r2c_16_input.csv", "./test_data/r2c_16_output.csv"},
        neo::fft::test_path{"./test_data/r2c_32_input.csv", "./test_data/r2c_32_output.csv"},
        neo::fft::test_path{"./test_data/r2c_64_input.csv", "./test_data/r2c_64_output.csv"},
        neo::fft::test_path{"./test_data/r2c_128_input.csv", "./test_data/r2c_128_output.csv"},
        neo::fft::test_path{"./test_data/r2c_512_input.csv", "./test_data/r2c_512_output.csv"}
    );

    auto const tc    = neo::fft::load_test_data<Float>(paths).value();
    auto const size  = tc.input.size();
    auto const order = neo::fft::ilog2(size);

    auto input  = std::vector<Float>(size_t(size), Float(0));
    auto output = std::vector<std::complex<Float>>(size_t(size / 2 + 1), Float(0));
    std::transform(tc.input.begin(), tc.input.end(), input.begin(), [](auto c) { return c.real(); });

    auto rfft = neo::fft::rfft_plan<Float>{order};
    rfft(
        Kokkos::mdspan{input.data(), Kokkos::extents{input.size()}},
        Kokkos::mdspan{output.data(), Kokkos::extents{output.size()}}
    );
    REQUIRE(neo::fft::allclose(tc.expected, output));
}

TEMPLATE_TEST_CASE("neo/fft/transform/rfft: roundtrip(rfft_plan)", "", float, double)
{
    using Float = TestType;

    auto order      = GENERATE(1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
    auto const size = 1UL << static_cast<std::size_t>(order);

    auto signal   = std::vector<Float>(size, Float(0));
    auto spectrum = std::vector<std::complex<Float>>(size / 2UL + 1UL, Float(0));

    auto rng  = std::mt19937{std::random_device{}()};
    auto dist = std::uniform_real_distribution<Float>{Float(-1), Float(1)};
    std::generate(signal.begin(), signal.end(), [&dist, &rng] { return dist(rng); });
    auto const original = signal;

    auto rfft = neo::fft::rfft_plan<Float>{static_cast<std::size_t>(order)};
    rfft(
        Kokkos::mdspan{signal.data(), Kokkos::extents{signal.size()}},
        Kokkos::mdspan{spectrum.data(), Kokkos::extents{spectrum.size()}}
    );
    rfft(
        Kokkos::mdspan{spectrum.data(), Kokkos::extents{spectrum.size()}},
        Kokkos::mdspan{signal.data(), Kokkos::extents{signal.size()}}
    );

    auto const scale = Float(1) / static_cast<Float>(size);
    std::transform(signal.begin(), signal.end(), signal.begin(), [scale](auto c) { return c * scale; });

    REQUIRE(neo::fft::allclose(original, signal));
}

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

    auto fft  = neo::fft::fft_plan<std::complex<Float>>{n};
    auto rfft = neo::fft::rfft_plan<Float>{order};

    auto a_rev = std::array<std::complex<Float>, n / 2 + 1>{};
    auto b_rev = std::array<std::complex<Float>, n / 2 + 1>{};
    rfft(
        Kokkos::mdspan{a.data(), Kokkos::extents{a.size()}},
        Kokkos::mdspan{a_rev.data(), Kokkos::extents{a_rev.size()}}
    );
    rfft(
        Kokkos::mdspan{b.data(), Kokkos::extents{b.size()}},
        Kokkos::mdspan{b_rev.data(), Kokkos::extents{b_rev.size()}}
    );

    auto inout = std::array<std::complex<Float>, n>{};
    std::transform(a.begin(), a.end(), b.begin(), inout.begin(), [](auto ra, auto rb) { return std::complex{ra, rb}; });

    fft(Kokkos::mdspan{inout.data(), Kokkos::extents{inout.size()}}, neo::fft::direction::forward);

    auto ca = std::array<std::complex<Float>, n / 2 + 1>{};
    auto cb = std::array<std::complex<Float>, n / 2 + 1>{};
    neo::fft::extract_two_real_dfts<Float>(inout, ca, cb);

    CHECK(neo::fft::allclose(a_rev, ca));
    CHECK(neo::fft::allclose(b_rev, cb));
}
