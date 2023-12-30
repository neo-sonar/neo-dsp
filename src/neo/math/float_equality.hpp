// SPDX-License-Identifier: MIT
#pragma once

#include <neo/complex/complex.hpp>

#include <algorithm>
#include <concepts>
#include <functional>

namespace neo::float_equality {

template<typename FloatOrComplex>
[[nodiscard]] auto exact(FloatOrComplex x, FloatOrComplex y) -> bool
{
    static constexpr auto const eq = std::equal_to{};

    if constexpr (complex<FloatOrComplex>) {
        return eq(x.real(), y.real()) and eq(x.imag(), y.imag());
    } else {
        return eq(x, y);
    }
}

}  // namespace neo::float_equality
