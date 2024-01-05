// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <cassert>
#include <concepts>

namespace neo {

template<std::integral Int>
[[nodiscard]] constexpr auto bit_log2(Int x) -> Int
{
    assert(x > Int(0));

    auto result = Int{0};
    for (; x > Int(1); x >>= Int(1)) {
        ++result;
    }
    return result;
}

}  // namespace neo
