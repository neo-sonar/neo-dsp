#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/scale.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/math/float_equality.hpp>

#include <algorithm>
#include <functional>
#include <limits>

namespace neo {

template<in_object InObj>
    requires std::floating_point<typename InObj::value_type>
[[nodiscard]] auto normalize_energy_factor(InObj obj) noexcept
{
    using Float = typename InObj::value_type;

    auto energy = Float(0);
    if constexpr (InObj::rank() == 1) {
        for (decltype(obj.extent(0)) i{0}; i < obj.extent(0); ++i) {
            energy += obj[i] * obj[i];
        }

    } else {
        for (decltype(obj.extent(0)) i{0}; i < obj.extent(0); ++i) {
            for (decltype(obj.extent(1)) j{0}; j < obj.extent(1); ++j) {
                energy += obj(i, j) * obj(i, j);
            }
        }
    }

    if (float_equality::exact(energy, Float(0))) {
        return Float(1);
    }

    return Float(1) / std::sqrt(energy);
}

// energy = sum(obj^2)
// return obj / sqrt(energy)
template<inout_object InOutObj>
auto normalize_energy(InOutObj obj) noexcept -> void
{
    scale(normalize_energy_factor(obj), obj);
}

}  // namespace neo
