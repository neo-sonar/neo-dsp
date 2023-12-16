#pragma once

#include <complex>
#include <concepts>
#include <cstdlib>
#include <type_traits>

namespace neo {

template<typename T>
inline constexpr auto const is_complex = false;

template<std::floating_point Float>
inline constexpr auto const is_complex<std::complex<Float>> = true;

template<typename T>
concept complex = is_complex<T>;

template<typename T>
concept float_or_complex = std::floating_point<T> or complex<T>;

}  // namespace neo
