#pragma once

#include <neo/container/mdspan.hpp>

#include <cmath>
#include <concepts>
#include <utility>

namespace neo {

template<std::floating_point Float>
[[nodiscard]] auto hertz_to_mel(Float hertz) noexcept -> Float
{
    return Float(2595) * std::log10(Float(1) + hertz / Float(700));
}

template<std::floating_point Float>
[[nodiscard]] auto mel_to_hertz(Float mels) noexcept -> Float
{
    return Float(700) * (std::pow(Float(10), mels / Float(2595)) - Float(1));
}

template<out_vector OutVec, std::floating_point Float>
auto mel_frequencies(OutVec out, Float fmin, Float fmax) noexcept -> void
{
    auto const n_mels = out.extent(0);
    if (std::cmp_equal(n_mels, 0)) {
        return;
    }

    auto const min_mel = hertz_to_mel(fmin);
    auto const max_mel = hertz_to_mel(fmax);
    auto const step    = (max_mel - min_mel) / static_cast<Float>(n_mels - 1);

    for (auto i{0}; std::cmp_less(i, n_mels); ++i) {
        out[i] = mel_to_hertz(min_mel + step * static_cast<Float>(i));
    }
}

}  // namespace neo
