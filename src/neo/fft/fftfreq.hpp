#pragma once

#include <neo/container/mdspan.hpp>

#include <concepts>

namespace neo {

template<std::floating_point T>
[[nodiscard]] constexpr auto fftfreq(std::integral auto size, std::integral auto index, double inverseSampleRate) -> T
{
    auto const fs      = T(1) / static_cast<T>(inverseSampleRate);
    auto const invSize = T(1) / static_cast<T>(size);
    return static_cast<T>(index) * fs * invSize;
}

template<out_vector Vec>
    requires(std::floating_point<typename Vec::value_type>)
constexpr auto fftfreq(Vec vec, double inverseSampleRate) noexcept -> void
{
    auto const size = static_cast<int>(vec.extent(0));
    for (auto i{0}; i < size; ++i) {
        vec[i] = fftfreq<typename Vec::value_type>(size, i, inverseSampleRate);
    }
}

}  // namespace neo
