#pragma once

#include <neo/fft/config.hpp>

#include <neo/fft/algorithm/scale.hpp>
#include <neo/fft/container/mdspan.hpp>
#include <neo/math/float_equality.hpp>

#include <algorithm>
#include <functional>
#include <limits>

namespace neo::fft {

template<in_object InObj>
    requires(std::floating_point<typename InObj::value_type>)
[[nodiscard]] auto peak_normalization_factor(InObj obj)
{
    using Float = typename InObj::value_type;

    auto absMax = Float(0);
    if constexpr (InObj::rank() == 1) {
        if (obj.extent(0) == 0) {
            return Float(1);
        }

        for (decltype(obj.extent(0)) i{0}; i < obj.extent(0); ++i) {
            absMax = std::max(absMax, std::abs(obj[i]));
        }

    } else {
        if (obj.extent(0) == 0 and obj.extent(1) == 0) {
            return Float(1);
        }

        for (decltype(obj.extent(0)) i{0}; i < obj.extent(0); ++i) {
            for (decltype(obj.extent(1)) j{0}; j < obj.extent(1); ++j) {
                absMax = std::max(absMax, std::abs(obj(i, j)));
            }
        }
    }

    if (float_equality::exact(absMax, Float(0))) {
        return Float(1);
    }

    return Float(1) / std::abs(absMax);
}

// normalized_sample = sample / max(abs(buffer))
template<inout_object InOutObj>
auto peak_normalize(InOutObj obj) -> void
{
    scale(peak_normalization_factor(obj), obj);
}

}  // namespace neo::fft
