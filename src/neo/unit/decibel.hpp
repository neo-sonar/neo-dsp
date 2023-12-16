#pragma once

#include <neo/math/fast_math.hpp>
#include <neo/math/precision.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>

namespace neo {

template<precision Precision = precision::estimate, std::floating_point Float>
[[nodiscard]] auto amplitude_to_db(Float gain, Float infinity = Float(-144)) noexcept -> Float
{
    if (gain <= Float(0)) {
        return infinity;
    }
    if constexpr (Precision == precision::accurate) {
        return (std::max)(infinity, Float(20) * std::log10(gain));
    } else {
        return (std::max)(infinity, Float(20) * static_cast<Float>(fast_log10(static_cast<float>(gain))));
    }
}

}  // namespace neo
