#pragma once

#include <algorithm>
#include <concepts>
#include <functional>

namespace neo::fft::float_equality {

template<typename FloatOrComplex>
[[nodiscard]] auto exact(FloatOrComplex x, FloatOrComplex y) -> bool
{
    static constexpr auto const eq = std::equal_to{};

    if constexpr (std::floating_point<FloatOrComplex>) {
        return eq(x, y);
    } else {
        return eq(x.real(), y.real()) and eq(x.imag(), y.imag());
    }
}

}  // namespace neo::fft::float_equality
