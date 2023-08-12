#pragma once

#include <concepts>

namespace neo {

template<std::integral T>
[[nodiscard]] constexpr auto power(T base, T exponent) -> T
{
    T result = 1;
    for (T i = 0; i < exponent; i++) {
        result *= base;
    }
    return result;
}

template<auto Base>
    requires(std::integral<decltype(Base)>)
[[nodiscard]] constexpr auto power(decltype(Base) exponent) -> decltype(Base)
{
    using Int = decltype(Base);

    if constexpr (Base == Int(2)) {
        return static_cast<Int>(Int(1) << exponent);
    } else {
        return power(Base, exponent);
    }
}

}  // namespace neo
