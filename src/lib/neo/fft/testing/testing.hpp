#pragma once

#include <neo/fft/config.hpp>

#include <neo/fft/container/mdspan.hpp>

#include <algorithm>
#include <complex>
#include <concepts>
#include <cstdio>
#include <filesystem>
#include <random>
#include <vector>

namespace neo::fft {

template<typename T>
using float_or_complex_value_type_t = decltype(std::abs(std::declval<T>()));

struct test_path
{
    std::filesystem::path input;
    std::filesystem::path expected;
};

template<std::floating_point Float>
struct test_data
{
    std::vector<std::complex<Float>> input;
    std::vector<std::complex<Float>> expected;
};

template<std::floating_point Float>
[[nodiscard]] auto load_test_data(test_path const& paths) -> test_data<Float>
{
    auto load_file = [](std::filesystem::path const& path) {
        auto* file = std::fopen(path.string().c_str(), "r");

        auto result = std::vector<std::complex<Float>>{};
        char line[512]{};
        while (std::fgets(line, sizeof(line), file)) {
            auto re = 0.0;
            auto im = 0.0;
            std::sscanf(line, "%lf,%lf\n", &re, &im);
            result.emplace_back(static_cast<Float>(re), static_cast<Float>(im));
        }

        std::fclose(file);

        return result;
    };

    return test_data<Float>{
        .input    = load_file(paths.input),
        .expected = load_file(paths.expected),
    };
}

template<typename FloatOrComplex>
[[nodiscard]] auto generate_noise_signal(std::size_t length, std::uint32_t seed)
{
    using Float = float_or_complex_value_type_t<FloatOrComplex>;

    auto rng    = std::mt19937{seed};
    auto dist   = std::uniform_real_distribution<Float>{Float(-1), Float(1)};
    auto signal = std::vector<FloatOrComplex>(length);

    if constexpr (std::floating_point<FloatOrComplex>) {
        std::generate(signal.begin(), signal.end(), [&] { return dist(rng); });
    } else {
        std::generate(signal.begin(), signal.end(), [&] { return FloatOrComplex{dist(rng), dist(rng)}; });
    }

    return signal;
}

template<std::floating_point Float>
[[nodiscard]] auto generate_identity_impulse(std::size_t blockSize, std::size_t numPartitions)
    -> KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<std::size_t, 2>>
{
    auto const windowSize = blockSize * 2;
    auto const numBins    = windowSize / 2 + 1;

    auto impulse = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<std::size_t, 2>>{
        numPartitions,
        numBins,
    };

    for (auto partition{0U}; partition < impulse.extent(0); ++partition) {
        for (auto bin{0U}; bin < impulse.extent(1); ++bin) {
            impulse(partition, bin) = std::complex{Float(1), Float(0)};
        }
    }
    return impulse;
}

}  // namespace neo::fft
