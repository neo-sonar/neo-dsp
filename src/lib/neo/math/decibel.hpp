#pragma once

#include <neo/math/fast_math.hpp>
#include <neo/math/math_precision.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>

namespace neo {

template<std::floating_point Float>
[[nodiscard]] auto to_decibels(Float gain, Float infinity, math_precision precision = math_precision::estimate) noexcept
    -> Float
{
    if (gain <= Float(0)) {
        return infinity;
    }
    if (precision == math_precision::estimate) {
        return (std::max)(infinity, Float(20) * static_cast<Float>(fast_log10(static_cast<float>(gain))));
    }

    return (std::max)(infinity, Float(20) * std::log10(gain));
}

template<std::floating_point Float>
[[nodiscard]] auto to_decibels(Float gain, math_precision precision = math_precision::estimate) noexcept -> Float
{
    return to_decibels(gain, Float(-144), precision);
}

}  // namespace neo
