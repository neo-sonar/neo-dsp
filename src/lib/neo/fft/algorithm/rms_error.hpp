#pragma once

#include <cmath>
#include <complex>
#include <concepts>
#include <functional>
#include <numeric>
#include <optional>
#include <span>

namespace neo::fft {

template<std::floating_point Float>
auto rms_error(std::span<Float const> original, std::span<Float const> reconstructed) noexcept -> std::optional<Float>
{
    if (original.empty()) { return std::nullopt; }
    if (original.size() != reconstructed.size()) { return std::nullopt; }

    auto diffSquared = [](Float x, Float y) {
        auto const diff = x - y;
        return diff * diff;
    };

    auto sum = std::transform_reduce(
        original.begin(),
        original.end(),
        reconstructed.begin(),
        Float(0),
        std::plus{},
        diffSquared
    );

    return std::sqrt(sum / static_cast<Float>(original.size()));
}

template<std::floating_point Float>
auto rms_error(
    std::span<std::complex<Float const>> original,
    std::span<std::complex<Float const>> reconstructed
) noexcept -> std::optional<Float>
{
    if (original.empty()) { return std::nullopt; }
    if (original.size() != reconstructed.size()) { return std::nullopt; }

    auto diffSquared = [](std::complex<Float> x, std::complex<Float> y) {
        auto const re = x.real() - y.real();
        auto const im = x.imag() - y.imag();
        return (re * re) + (im * im);
    };

    auto sum = std::transform_reduce(
        original.begin(),
        original.end(),
        reconstructed.begin(),
        Float(0),
        std::plus{},
        diffSquared
    );

    return std::sqrt(sum / static_cast<Float>(original.size()));
}

}  // namespace neo::fft
