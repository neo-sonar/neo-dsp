// SPDX-License-Identifier: MIT
#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/mean.hpp>
#include <neo/container/mdspan.hpp>

#include <optional>
#include <utility>

namespace neo {

template<in_object InObj>
[[nodiscard]] constexpr auto variance(InObj x) noexcept -> std::optional<typename InObj::value_type>
{
    using Float = typename InObj::value_type;

    if (std::cmp_less(x.size(), 2)) {
        return std::nullopt;
    }

    auto const avg = mean_unchecked(x);

    auto deviation = Float(0);

    if constexpr (InObj::rank() == 1) {
        for (auto i{0}; std::cmp_less(i, x.extent(0)); ++i) {
            auto const diff = x[i] - avg;
            deviation += diff * diff;
        }
    } else {
        for (auto i{0}; std::cmp_less(i, x.extent(0)); ++i) {
            for (auto j{0}; std::cmp_less(j, x.extent(1)); ++j) {
                auto const diff = x(i, j) - avg;
                deviation += diff * diff;
            }
        }
    }

    return deviation / static_cast<Float>(x.size());
}

}  // namespace neo
