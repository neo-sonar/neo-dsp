#pragma once

namespace neo::fft {

template<typename T>
[[nodiscard]] constexpr auto power(T base, T exponent) -> T
{
    T result = 1;
    for (T i = 0; i < exponent; i++) {
        result *= base;
    }
    return result;
}

template<auto Base>
[[nodiscard]] constexpr auto power(decltype(Base) exponent) -> decltype(Base)
{
    return power(Base, exponent);
}

}  // namespace neo::fft
