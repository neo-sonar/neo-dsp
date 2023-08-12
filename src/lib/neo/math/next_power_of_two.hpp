#pragma once

#include <bit>
#include <concepts>

namespace neo {

namespace detail {

template<std::unsigned_integral Integral>
[[nodiscard]] constexpr auto next_power_of_two_fallback(Integral x) noexcept -> Integral
{
    --x;
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return x + 1;
}

}  // namespace detail

template<std::unsigned_integral Integral>
[[nodiscard]] constexpr auto next_power_of_two(Integral x) noexcept -> Integral
{
#if defined(__cpp_lib_int_pow2)
    return std::bit_ceil(x);
#else
    return detail::next_power_of_two_fallback(x);
#endif
}

}  // namespace neo
