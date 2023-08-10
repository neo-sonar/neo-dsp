#pragma once

#include <concepts>

namespace neo::fft {

template<std::floating_point T>
[[nodiscard]] constexpr auto fftfreq(std::integral auto windowSize, std::integral auto index, double sampleRate) -> T
{
    return static_cast<T>(index) * static_cast<T>(sampleRate) / static_cast<T>(windowSize);
}

}  // namespace neo::fft
