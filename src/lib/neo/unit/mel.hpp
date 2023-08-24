#pragma once

#include <cmath>
#include <concepts>

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

}  // namespace neo
