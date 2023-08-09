#include "test.hpp"

#include "doctest/doctest.h"
#include "neo/fft.hpp"

#include <array>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

static auto test_N8()
{
    auto const dft = [] {
        auto in  = std::vector<std::complex<double>>{0, 1, 2, 3, 4, 5, 6, 7};
        auto out = std::vector<std::complex<double>>{0, 0, 0, 0, 0, 0, 0, 0};
        neo::fft::dft<double>(in, out);
        return out;
    }();

    REQUIRE(neo::fft::almost_equal(dft[0], std::complex(28.0, 0.0)));
    REQUIRE(neo::fft::almost_equal(dft[1], std::complex(-4.000000, 9.656854)));
    REQUIRE(neo::fft::almost_equal(dft[2], std::complex(-4.000000, 4.000000)));
    REQUIRE(neo::fft::almost_equal(dft[3], std::complex(-4.000000, 1.656854)));
    REQUIRE(neo::fft::almost_equal(dft[4], std::complex(-4.000000, 0.000000)));
    REQUIRE(neo::fft::almost_equal(dft[5], std::complex(-4.000000, -1.656854)));
    REQUIRE(neo::fft::almost_equal(dft[6], std::complex(-4.000000, -4.000000)));
    REQUIRE(neo::fft::almost_equal(dft[7], std::complex(-4.000000, -9.656854)));

    auto const a = [] {
        auto tw  = neo::fft::twiddle_table_radix2<std::complex<double>, 8>();
        auto buf = std::vector<std::complex<double>>{0, 1, 2, 3, 4, 5, 6, 7};
        neo::fft::c2c_radix2(std::span{buf}, tw);
        return buf;
    }();

    auto const b = [] {
        auto tw  = neo::fft::twiddle_table_radix2<std::complex<double>, 8>();
        auto buf = std::vector<std::complex<double>>{0, 1, 2, 3, 4, 5, 6, 7};
        neo::fft::c2c_radix2_alt(std::span{buf}, tw);
        return buf;
    }();

    auto const c = [] {
        auto buf = std::vector<std::complex<double>>{0, 1, 2, 3, 4, 5, 6, 7};
        auto eng = neo::fft::c2c_radix2_plan<std::complex<double>>{buf.size()};
        eng(buf, neo::fft::direction::forward);
        return buf;
    }();

    REQUIRE(neo::fft::allclose(dft, a));
    REQUIRE(neo::fft::allclose(dft, b));
    REQUIRE(neo::fft::allclose(dft, c));
}

static auto run_c2c_test(auto const& paths)
{
    auto const testCase = load_test_data(paths).value();

    {
        auto in  = testCase.input;
        auto out = std::vector<std::complex<double>>(in.size());
        neo::fft::dft<double>(in, out);
        REQUIRE(neo::fft::allclose(testCase.expected, out));
    }

    {
        auto inout = testCase.input;
        auto tw    = neo::fft::twiddle_table_radix2<std::complex<double>>(inout.size());
        neo::fft::c2c_radix2(std::span{inout}, tw);
        REQUIRE(neo::fft::allclose(testCase.expected, inout));
    }

    {
        auto inout = testCase.input;
        auto tw    = neo::fft::twiddle_table_radix2<std::complex<double>>(inout.size());
        neo::fft::c2c_radix2_alt(std::span{inout}, tw);
        REQUIRE(neo::fft::allclose(testCase.expected, inout));
    }

    {
        auto inout = testCase.input;
        auto eng   = neo::fft::c2c_radix2_plan<std::complex<double>>{inout.size()};
        eng(inout, neo::fft::direction::forward);
        REQUIRE(neo::fft::allclose(testCase.expected, inout));
    }
}

static auto run_c2c_roundtrip_test(std::size_t size) -> void
{
    auto const n        = size;
    auto const twiddles = neo::fft::twiddle_table_radix2<std::complex<double>>(n);

    auto buffer = std::vector<std::complex<double>>(n, std::complex<double>(0));
    auto rng    = std::mt19937{std::random_device{}()};
    auto dist   = std::uniform_real_distribution<double>{-1.0, 1.0};
    std::generate(buffer.begin(), buffer.end(), [&dist, &rng] { return std::complex<double>{dist(rng), dist(rng)}; });

    auto inout = buffer;
    auto c2c   = neo::fft::c2c_radix2_plan<std::complex<double>>{n};
    c2c(inout, neo::fft::direction::forward);
    c2c(inout, neo::fft::direction::backward);
    std::transform(inout.begin(), inout.end(), inout.begin(), [n](auto c) { return c / static_cast<double>(n); });

    REQUIRE(neo::fft::allclose(buffer, inout));
}

TEST_CASE("fft/radix2: c2c")
{
    test_N8();

    auto const tests = std::array{
        test_path{  "./test_data/c2c_8_input.csv",   "./test_data/c2c_8_output.csv"},
        test_path{ "./test_data/c2c_16_input.csv",  "./test_data/c2c_16_output.csv"},
        test_path{ "./test_data/c2c_32_input.csv",  "./test_data/c2c_32_output.csv"},
        test_path{ "./test_data/c2c_16_input.csv",  "./test_data/c2c_16_output.csv"},
        test_path{ "./test_data/c2c_32_input.csv",  "./test_data/c2c_32_output.csv"},
        test_path{ "./test_data/c2c_64_input.csv",  "./test_data/c2c_64_output.csv"},
        test_path{"./test_data/c2c_128_input.csv", "./test_data/c2c_128_output.csv"},
        test_path{"./test_data/c2c_512_input.csv", "./test_data/c2c_512_output.csv"},
    };

    for (auto const& tc : tests) {
        CAPTURE(tc);
        run_c2c_test(tc);
    }

    for (auto size : std::array{8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}) {
        CAPTURE(size);
        run_c2c_roundtrip_test(size_t(size));
    }
}

static auto run_r2c_test(auto const& paths) -> void
{
    auto const tc    = load_test_data(paths).value();
    auto const size  = tc.input.size();
    auto const order = neo::fft::ilog2(size);

    auto input  = std::vector<double>(size_t(size), 0.0);
    auto output = std::vector<std::complex<double>>(size_t(size / 2 + 1), 0.0);
    std::transform(tc.input.begin(), tc.input.end(), input.begin(), [](auto c) { return c.real(); });

    auto rfft = neo::fft::rfft_radix2_plan<double>{order};
    rfft(input, output);
    REQUIRE(neo::fft::allclose(tc.expected, output));
}

static auto run_rfft_roundtrip_test(std::size_t order) -> void
{
    auto const size = 1 << order;

    auto signal   = std::vector<double>(size, double(0));
    auto spectrum = std::vector<std::complex<double>>(size / 2 + 1, 0.0);

    auto rng  = std::mt19937{std::random_device{}()};
    auto dist = std::uniform_real_distribution<double>{-1.0, 1.0};
    std::generate(signal.begin(), signal.end(), [&dist, &rng] { return dist(rng); });
    auto const original = signal;

    auto rfft = neo::fft::rfft_radix2_plan<double>{order};
    rfft(signal, spectrum);
    rfft(spectrum, signal);
    std::transform(signal.begin(), signal.end(), signal.begin(), [size](auto c) {
        return c / static_cast<double>(size);
    });

    REQUIRE(neo::fft::allclose(original, signal));
}

TEST_CASE("fft/radix2: r2c")
{
    for (auto const& tc : std::array{
             test_path{  "./test_data/r2c_8_input.csv",   "./test_data/r2c_8_output.csv"},
             test_path{ "./test_data/r2c_16_input.csv",  "./test_data/r2c_16_output.csv"},
             test_path{ "./test_data/r2c_32_input.csv",  "./test_data/r2c_32_output.csv"},
             test_path{ "./test_data/r2c_16_input.csv",  "./test_data/r2c_16_output.csv"},
             test_path{ "./test_data/r2c_32_input.csv",  "./test_data/r2c_32_output.csv"},
             test_path{ "./test_data/r2c_64_input.csv",  "./test_data/r2c_64_output.csv"},
             test_path{"./test_data/r2c_128_input.csv", "./test_data/r2c_128_output.csv"},
             test_path{"./test_data/r2c_512_input.csv", "./test_data/r2c_512_output.csv"},
    }) {
        CAPTURE(tc);
        run_r2c_test(tc);
    }

    for (auto order : std::array{1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12}) {
        CAPTURE(order);
        run_rfft_roundtrip_test(size_t(order));
    }
}

TEST_CASE("fft/rfft: extract_two_real_dfts")
{
    static constexpr auto n     = 8UL;
    static constexpr auto order = neo::fft::ilog2(n);

    auto rng    = std::mt19937{std::random_device{}()};
    auto dist   = std::uniform_real_distribution<float>{-1, 1};
    auto random = [&dist, &rng] { return dist(rng); };

    auto a = std::array<float, n>{};
    auto b = std::array<float, n>{};
    std::generate(a.begin(), a.end(), random);
    std::generate(b.begin(), b.end(), random);

    auto fft  = neo::fft::c2c_radix2_plan<std::complex<float>>{n};
    auto rfft = neo::fft::rfft_radix2_plan<float>{order};

    auto a_rev = std::array<std::complex<float>, n / 2 + 1>{};
    auto b_rev = std::array<std::complex<float>, n / 2 + 1>{};
    rfft(a, a_rev);
    rfft(b, b_rev);

    auto inout = std::array<std::complex<float>, n>{};
    std::transform(a.begin(), a.end(), b.begin(), inout.begin(), [](auto ra, auto rb) { return std::complex{ra, rb}; });

    fft(inout, neo::fft::direction::forward);

    auto ca = std::array<std::complex<float>, n / 2 + 1>{};
    auto cb = std::array<std::complex<float>, n / 2 + 1>{};
    neo::fft::extract_two_real_dfts<float>(inout, ca, cb);

    CHECK(neo::fft::allclose(a_rev, ca));
    CHECK(neo::fft::allclose(b_rev, cb));
}
