#pragma once

#include <concepts>

namespace neo {

template<std::floating_point T>
[[nodiscard]] constexpr auto fftfreq(std::integral auto size, std::integral auto index, double inverseSampleRate) -> T
{
    return static_cast<T>(index) / static_cast<T>(inverseSampleRate) / static_cast<T>(size);
}

}  // namespace neo
