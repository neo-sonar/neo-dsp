// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>

namespace neo {

template<std::integral Int>
[[nodiscard]] constexpr auto is_even(Int x) noexcept -> bool
{
    return (x & Int(1)) == 0;
}

}  // namespace neo
