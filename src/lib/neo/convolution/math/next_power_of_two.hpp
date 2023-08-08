#pragma once

#include <bit>
#include <concepts>

namespace neo::fft {

template<std::unsigned_integral Integral>
[[nodiscard]] constexpr auto next_power_of_two(Integral x) noexcept -> Integral
{

#if defined(__cpp_lib_int_pow2)
    return std::bit_ceil(x);
#else
    --x;
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return x + 1;
#endif
}

}  // namespace neo::fft

static_assert(neo::fft::next_power_of_two(1U) == 1U);
static_assert(neo::fft::next_power_of_two(2U) == 2U);
static_assert(neo::fft::next_power_of_two(3U) == 4U);
static_assert(neo::fft::next_power_of_two(4U) == 4U);
static_assert(neo::fft::next_power_of_two(100U) == 128U);