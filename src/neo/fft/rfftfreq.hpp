// SPDX-License-Identifier: MIT

#pragma once

#include <neo/container/mdspan.hpp>

#include <concepts>

namespace neo {

/// \ingroup neo-fft
template<std::floating_point T>
[[nodiscard]] constexpr auto rfftfreq(std::integral auto size, std::integral auto index, double inv_sample_rate) -> T
{
    auto const fs       = T(1) / static_cast<T>(inv_sample_rate);
    auto const inv_size = T(1) / static_cast<T>(size);
    return static_cast<T>(index) * fs * inv_size;
}

/// \ingroup neo-fft
template<out_vector Vec>
    requires(std::floating_point<typename Vec::value_type>)
constexpr auto rfftfreq(Vec vec, double inv_sample_rate) noexcept -> void
{
    auto const size = static_cast<int>(vec.extent(0));
    for (auto i{0}; i < size; ++i) {
        vec[i] = rfftfreq<typename Vec::value_type>(size, i, inv_sample_rate);
    }
}

}  // namespace neo
