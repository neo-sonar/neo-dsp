#pragma once

#include <complex>

namespace neo::fft {

template<typename T>
inline constexpr auto const is_complex = false;

template<typename T>
inline constexpr auto const is_complex<std::complex<T>> = true;

}  // namespace neo::fft
