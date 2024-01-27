// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>

namespace neo {

/// \ingroup neo-math
template<std::integral Int>
[[nodiscard]] constexpr auto is_odd(Int x) noexcept -> bool
{
    return (x & Int(1)) != 0;
}

}  // namespace neo
