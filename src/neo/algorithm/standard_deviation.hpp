// SPDX-License-Identifier: MIT
#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/variance.hpp>
#include <neo/container/mdspan.hpp>

#include <cmath>
#include <optional>

namespace neo {

template<in_object InObj>
[[nodiscard]] constexpr auto standard_deviation(InObj x) noexcept -> std::optional<typename InObj::value_type>
{
    if (auto const var = variance(x); var.has_value()) {
        return std::sqrt(*var);
    }
    return std::nullopt;
}

}  // namespace neo
