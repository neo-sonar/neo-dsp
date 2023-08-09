#pragma once

#include <numbers>
#include <type_traits>

namespace neo::fft {

template<typename T>
[[nodiscard]] constexpr auto power(T base, T exponent) -> T
{
    T result = 1;
    for (T i = 0; i < exponent; i++) { result *= base; }
    return result;
}

template<auto Base>
[[nodiscard]] constexpr auto power(decltype(Base) exponent) -> decltype(Base)
{
    return power(Base, exponent);
}

template<typename Unsigned>
[[nodiscard]] constexpr auto ilog2(Unsigned num) noexcept -> Unsigned
{
    static_assert(std::is_unsigned_v<Unsigned>);

    auto result = Unsigned{0};
    for (; num > 1; num >>= 1) { ++result; }
    return result;
}

}  // namespace neo::fft
