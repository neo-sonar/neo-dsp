#pragma once

#include <concepts>

namespace neo::fft {

template<std::integral T>
[[nodiscard]] constexpr auto divide_round_up(T x, T y) noexcept -> T
{
    return (x + y - 1) / y;
}

}  // namespace neo::fft
