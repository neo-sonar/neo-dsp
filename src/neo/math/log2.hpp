// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <concepts>

namespace neo::math {

namespace detail {

using std::log2;

auto log2(auto const&) -> void = delete;

template<typename T>
concept has_adl_log2 = requires(T const& t) { log2(t); };

struct log2_fn
{
    template<std::integral T>
    [[nodiscard]] constexpr auto operator()(T x) const noexcept
    {
        return std::log2(x);
    }

    template<has_adl_log2 T>
        requires(not std::integral<T>)
    [[nodiscard]] constexpr auto operator()(T x) const noexcept
    {
        return log2(x);
    }
};

}  // namespace detail

inline constexpr auto const log2 = detail::log2_fn{};

}  // namespace neo::math
