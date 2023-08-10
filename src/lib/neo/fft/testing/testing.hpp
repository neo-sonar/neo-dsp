#pragma once

#include <neo/fft/container/mdspan.hpp>

#include <complex>
#include <concepts>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <random>
#include <vector>

namespace neo::fft {
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
[[nodiscard]] auto load_test_data_file(std::filesystem::path const& path)
    -> std::optional<std::vector<std::complex<Float>>>
{
    if (not std::filesystem::exists(path)) {
        return {};
    }
    if (not std::filesystem::is_regular_file(path)) {
        return {};
    }

    auto* file = std::fopen(path.string().c_str(), "r");
    if (file == nullptr) {
        return {};
    }

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
}

template<std::floating_point Float>
[[nodiscard]] auto load_test_data(test_path const& paths) -> std::optional<test_data<Float>>
{
    auto input = load_test_data_file<Float>(paths.input);
    if (not input) {
        return {};
    }

    auto expected = load_test_data_file<Float>(paths.expected);
    if (not expected) {
        return {};
    }

    return test_data<Float>{
        .input    = std::move(*input),
        .expected = std::move(*expected),
    };
}

template<std::floating_point Float>
[[nodiscard]] auto make_noise_signal(std::size_t length, std::uint32_t seed) -> std::vector<Float>
{
    auto rng    = std::mt19937{seed};
    auto dist   = std::uniform_real_distribution<Float>{Float(-1), Float(1)};
    auto signal = std::vector<Float>(length, Float(0));
    std::generate(signal.begin(), signal.end(), [&] { return dist(rng); });
    return signal;
}

template<std::floating_point Float>
[[nodiscard]] auto make_identity_impulse(std::size_t blockSize, std::size_t numPartitions)
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
