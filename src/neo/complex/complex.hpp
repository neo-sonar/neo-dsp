// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <concepts>

namespace neo {

template<typename T>
inline constexpr auto const is_complex = false;

template<std::floating_point Float>
inline constexpr auto const is_complex<std::complex<Float>> = true;

template<typename T>
concept complex = is_complex<T>;

}  // namespace neo
