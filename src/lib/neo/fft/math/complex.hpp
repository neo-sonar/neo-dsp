#pragma once

#include <complex>
#include <concepts>
#include <cstdlib>
#include <type_traits>

namespace neo::fft {

template<typename T>
inline constexpr auto const is_complex = false;

template<std::floating_point Float>
inline constexpr auto const is_complex<std::complex<Float>> = true;

template<typename T>
concept complex = is_complex<T>;

template<typename T>
concept float_or_complex = std::floating_point<T> or is_complex<T>;

template<typename RealOrComplex>
using real_or_complex_value_t = decltype([] {
    if constexpr (std::floating_point<RealOrComplex>) {
        return RealOrComplex{};
    } else {
        static_assert(is_complex<RealOrComplex>);
        return typename RealOrComplex::value_type{};
    }
}());

}  // namespace neo::fft
