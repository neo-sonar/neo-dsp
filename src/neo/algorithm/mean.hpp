// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/type_traits/value_type_t.hpp>

#include <optional>
#include <utility>

namespace neo {

template<in_object InObj>
[[nodiscard]] constexpr auto mean_unchecked(InObj x) noexcept -> value_type_t<InObj>
{
    using Float = value_type_t<InObj>;

    auto sum = Float(0);

    if constexpr (InObj::rank() == 1) {
        for (auto i{0}; std::cmp_less(i, x.extent(0)); ++i) {
            sum += x[i];
        }
    } else {
        for (auto i{0}; std::cmp_less(i, x.extent(0)); ++i) {
            for (auto j{0}; std::cmp_less(j, x.extent(1)); ++j) {
                sum += x(i, j);
            }
        }
    }

    return sum / static_cast<Float>(x.size());
}

template<in_object InObj>
[[nodiscard]] constexpr auto mean(InObj x) noexcept -> std::optional<value_type_t<InObj>>
{
    if (std::cmp_less(x.size(), 1)) {
        return std::nullopt;
    }
    return mean_unchecked(x);
}

}  // namespace neo
