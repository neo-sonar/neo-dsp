#include "dft.hpp"
#include "radix2.hpp"

#include <neo/fft/algorithm/allclose.hpp>
#include <neo/fft/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <random>
#include <vector>

TEMPLATE_TEST_CASE("neo/fft/transform/radix2: make_radix2_twiddles", "", float, double)
{
    using Float   = TestType;
    using Complex = std::complex<Float>;

    auto const array  = neo::fft::make_radix2_twiddles<Complex, 64>();
    auto const vector = neo::fft::make_radix2_twiddles<Complex>(64);

    for (auto i{0UL}; i < array.size(); ++i) {
        CAPTURE(i);
        REQUIRE(array[i].real() == Catch::Approx(vector[i].real()));
        REQUIRE(array[i].imag() == Catch::Approx(vector[i].imag()));
    }
}

TEMPLATE_TEST_CASE("neo/fft/transform/radix2: test_path(c2c)", "", double)
{
    using Float = TestType;

    auto paths = GENERATE(
        neo::fft::test_path{"./test_data/c2c_8_input.csv", "./test_data/c2c_8_output.csv"},
        neo::fft::test_path{"./test_data/c2c_16_input.csv", "./test_data/c2c_16_output.csv"},
        neo::fft::test_path{"./test_data/c2c_32_input.csv", "./test_data/c2c_32_output.csv"},
        neo::fft::test_path{"./test_data/c2c_16_input.csv", "./test_data/c2c_16_output.csv"},
        neo::fft::test_path{"./test_data/c2c_32_input.csv", "./test_data/c2c_32_output.csv"},
        neo::fft::test_path{"./test_data/c2c_64_input.csv", "./test_data/c2c_64_output.csv"},
        neo::fft::test_path{"./test_data/c2c_128_input.csv", "./test_data/c2c_128_output.csv"},
        neo::fft::test_path{"./test_data/c2c_512_input.csv", "./test_data/c2c_512_output.csv"}
    );

    auto const testCase = neo::fft::load_test_data<Float>(paths).value();

    {
        auto in  = testCase.input;
        auto out = std::vector<std::complex<Float>>(in.size());

        auto inVec  = Kokkos::mdspan{in.data(), Kokkos::extents{in.size()}};
        auto outVec = Kokkos::mdspan{out.data(), Kokkos::extents{out.size()}};
        neo::fft::dft(inVec, outVec);

        REQUIRE(neo::fft::allclose(testCase.expected, out));
    }

    {
        auto inout = testCase.input;
        auto tw    = neo::fft::make_radix2_twiddles<std::complex<Float>>(inout.size());
        neo::fft::c2c_radix2(Kokkos::mdspan{inout.data(), Kokkos::extents{inout.size()}}, tw);
        REQUIRE(neo::fft::allclose(testCase.expected, inout));
    }

    {
        auto inout = testCase.input;
        auto tw    = neo::fft::make_radix2_twiddles<std::complex<Float>>(inout.size());
        neo::fft::c2c_radix2_alt(Kokkos::mdspan{inout.data(), Kokkos::extents{inout.size()}}, tw);
        REQUIRE(neo::fft::allclose(testCase.expected, inout));
    }

    {
        auto inout = testCase.input;
        auto eng   = neo::fft::c2c_radix2_plan<std::complex<Float>>{inout.size()};
        eng(Kokkos::mdspan{inout.data(), Kokkos::extents{inout.size()}}, neo::fft::direction::forward);
        REQUIRE(neo::fft::allclose(testCase.expected, inout));
    }
}

TEMPLATE_TEST_CASE("neo/fft/transform/radix2: roundtrip(c2c)", "", float, double)
{
    using Float = TestType;

    auto size           = GENERATE(8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384);
    auto const twiddles = neo::fft::make_radix2_twiddles<std::complex<Float>>(static_cast<std::size_t>(size));

    auto buffer = std::vector<std::complex<Float>>(static_cast<std::size_t>(size), std::complex<Float>(0));
    auto rng    = std::mt19937{std::random_device{}()};
    auto dist   = std::uniform_real_distribution<Float>{Float(-1.0), Float(1.0)};
    std::generate(buffer.begin(), buffer.end(), [&dist, &rng] { return std::complex<Float>{dist(rng), dist(rng)}; });

    auto inout = buffer;
    auto c2c   = neo::fft::c2c_radix2_plan<std::complex<Float>>{static_cast<std::size_t>(size)};
    c2c(Kokkos::mdspan{inout.data(), Kokkos::extents{inout.size()}}, neo::fft::direction::forward);
    c2c(Kokkos::mdspan{inout.data(), Kokkos::extents{inout.size()}}, neo::fft::direction::backward);
    std::transform(inout.begin(), inout.end(), inout.begin(), [size](auto c) { return c / static_cast<Float>(size); });

    REQUIRE(neo::fft::allclose(buffer, inout));
}

TEMPLATE_TEST_CASE("neo/fft/transform/radix2: test_data(r2c)", "", float, double)
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

    auto rfft = neo::fft::rfft_radix2_plan<Float>{order};
    rfft(
        Kokkos::mdspan{input.data(), Kokkos::extents{input.size()}},
        Kokkos::mdspan{output.data(), Kokkos::extents{output.size()}}
    );
    REQUIRE(neo::fft::allclose(tc.expected, output));
}

TEMPLATE_TEST_CASE("neo/fft/transform/radix2: roundtrip(r2c)", "", float, double)
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

    REQUIRE(neo::fft::allclose(original, signal));
}
