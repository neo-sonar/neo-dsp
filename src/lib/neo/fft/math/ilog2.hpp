#pragma once

#include <concepts>
#include <numbers>

namespace neo::fft {

template<std::unsigned_integral Unsigned>
[[nodiscard]] constexpr auto ilog2(Unsigned num) noexcept -> Unsigned
{
    auto result = Unsigned{0};
    for (; num > 1; num >>= 1) {
        ++result;
    }
    return result;
}

}  // namespace neo::fft
