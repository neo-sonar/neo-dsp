#include "dft.hpp"
#include "radix2.hpp"

#include <neo/fft/algorithm/allclose.hpp>
#include <neo/fft/algorithm/scale.hpp>
#include <neo/fft/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
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

    auto const testCase = neo::fft::load_test_data<Float>(paths);
    auto const expected = Kokkos::mdspan{testCase.expected.data(), Kokkos::extents{testCase.expected.size()}};

    {
        auto in  = testCase.input;
        auto out = std::vector<std::complex<Float>>(in.size());

        auto inVec  = Kokkos::mdspan{in.data(), Kokkos::extents{in.size()}};
        auto outVec = Kokkos::mdspan{out.data(), Kokkos::extents{out.size()}};
        neo::fft::dft(inVec, outVec);

        REQUIRE(neo::fft::allclose(expected, outVec));
    }

    {
        auto inout = testCase.input;
        auto io    = Kokkos::mdspan{inout.data(), Kokkos::extents{inout.size()}};
        auto tw    = neo::fft::make_radix2_twiddles<std::complex<Float>>(inout.size());
        neo::fft::c2c_radix2(io, tw);
        REQUIRE(neo::fft::allclose(expected, io));
    }

    {
        auto inout = testCase.input;
        auto io    = Kokkos::mdspan{inout.data(), Kokkos::extents{inout.size()}};
        auto tw    = neo::fft::make_radix2_twiddles<std::complex<Float>>(inout.size());
        neo::fft::c2c_radix2_alt(io, tw);
        REQUIRE(neo::fft::allclose(expected, io));
    }

    {
        auto inout = testCase.input;
        auto io    = Kokkos::mdspan{inout.data(), Kokkos::extents{inout.size()}};
        auto plan  = neo::fft::fft_radix2_plan<std::complex<Float>>{inout.size()};
        plan(io, neo::fft::direction::forward);
        REQUIRE(neo::fft::allclose(expected, io));
    }
}

TEMPLATE_TEST_CASE("neo/fft/transform/radix2: roundtrip(fft_radix2_plan)", "", float, double)
{
    using Float = TestType;

    auto const size = GENERATE(as<size_t>{}, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384);

    auto const original = neo::fft::generate_noise_signal<std::complex<Float>>(size, Catch::getSeed());
    auto const noise    = Kokkos::mdspan{original.data(), Kokkos::extents{original.size()}};

    auto copy = original;
    auto io   = Kokkos::mdspan{copy.data(), Kokkos::extents{copy.size()}};

    auto plan = neo::fft::fft_radix2_plan<std::complex<Float>>{size};
    REQUIRE(plan.size() == size);
    REQUIRE(plan.order() == neo::fft::ilog2(size));

    plan(io, neo::fft::direction::forward);
    plan(io, neo::fft::direction::backward);
    neo::fft::scale(Float(1) / static_cast<Float>(plan.size()), io);

    REQUIRE(neo::fft::allclose(noise, io));
}
