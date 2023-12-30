// SPDX-License-Identifier: MIT
#pragma once

#include <concepts>

namespace neo {

template<std::integral T>
[[nodiscard]] constexpr auto idiv(T x, T y) noexcept -> T
{
    return (x + y - 1) / y;
}

}  // namespace neo
