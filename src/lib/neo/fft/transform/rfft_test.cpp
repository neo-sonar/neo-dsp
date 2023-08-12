#include "rfft.hpp"

#include <neo/fft/algorithm/allclose.hpp>
#include <neo/fft/testing/testing.hpp>
#include <neo/fft/transform/radix2.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <array>
#include <complex>
#include <random>

TEMPLATE_TEST_CASE("neo/fft/transform/rfft: test_data(rfft_radix2_plan)", "", float, double)
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

    auto const tc    = neo::fft::load_test_data<Float>(paths);
    auto const size  = tc.input.size();
    auto const order = neo::fft::ilog2(size);

    auto input  = std::vector<Float>(size_t(size), Float(0));
    auto output = std::vector<std::complex<Float>>(size_t(size / 2 + 1), Float(0));
    std::transform(tc.input.begin(), tc.input.end(), input.begin(), [](auto c) { return c.real(); });

    auto in  = Kokkos::mdspan{input.data(), Kokkos::extents{input.size()}};
    auto out = Kokkos::mdspan{output.data(), Kokkos::extents{output.size()}};

    auto rfft = neo::fft::rfft_radix2_plan<Float>{order};
    rfft(in, out);

    auto const expected = Kokkos::mdspan{tc.expected.data(), Kokkos::extents{tc.expected.size()}};
    REQUIRE(neo::fft::allclose(expected, out));
}

TEMPLATE_TEST_CASE("neo/fft/transform/rfft: roundtrip(rfft_radix2_plan)", "", float, double)
{
    using Float = TestType;

    auto order      = GENERATE(1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
    auto const size = 1UL << static_cast<std::size_t>(order);

    auto signal         = neo::fft::generate_noise_signal<Float>(size, Catch::getSeed());
    auto spectrum       = std::vector<std::complex<Float>>(size / 2UL + 1UL, Float(0));
    auto const original = signal;

    auto rfft = neo::fft::rfft_radix2_plan<Float>{static_cast<std::size_t>(order)};
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

    REQUIRE(neo::fft::allclose(
        Kokkos::mdspan{original.data(), Kokkos::extents{original.size()}},
        Kokkos::mdspan{signal.data(), Kokkos::extents{signal.size()}}
    ));
}

TEMPLATE_TEST_CASE("neo/fft/transform/rfft: extract_two_real_dfts", "", float, double)
{
    using Float = TestType;

    auto order           = GENERATE(as<std::size_t>{}, 4, 5, 6, 7, 8);
    auto const size      = std::size_t(1) << order;
    auto const numCoeffs = size / 2 + 1;
    CAPTURE(order);
    CAPTURE(size);
    CAPTURE(numCoeffs);

    auto rfft = neo::fft::rfft_radix2_plan<Float>{order};
    auto fft  = neo::fft::fft_radix2_plan<std::complex<Float>>{size};

    auto rng    = std::mt19937{Catch::getSeed()};
    auto dist   = std::uniform_real_distribution<Float>{Float(-1), Float(1)};
    auto random = [&dist, &rng] { return dist(rng); };

    auto a = KokkosEx::mdarray<Float, Kokkos::dextents<size_t, 1>>{size};
    auto b = KokkosEx::mdarray<Float, Kokkos::dextents<size_t, 1>>{size};
    std::generate(a.data(), a.data() + a.size(), random);
    std::generate(b.data(), b.data() + b.size(), random);

    auto a_rev = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>>{numCoeffs};
    auto b_rev = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>>{numCoeffs};
    rfft(a.to_mdspan(), a_rev.to_mdspan());
    rfft(b.to_mdspan(), b_rev.to_mdspan());

    auto inout = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>>{size};
    auto merge = [](Float ra, Float rb) { return std::complex{ra, rb}; };
    std::transform(a.data(), a.data() + a.size(), b.data(), inout.data(), merge);

    fft(inout.to_mdspan(), neo::fft::direction::forward);

    auto ca = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>>{numCoeffs};
    auto cb = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>>{numCoeffs};
    neo::fft::extract_two_real_dfts(inout.to_mdspan(), ca.to_mdspan(), cb.to_mdspan());

    REQUIRE(neo::fft::allclose(a_rev.to_mdspan(), ca.to_mdspan()));
    REQUIRE(neo::fft::allclose(b_rev.to_mdspan(), cb.to_mdspan()));
}
