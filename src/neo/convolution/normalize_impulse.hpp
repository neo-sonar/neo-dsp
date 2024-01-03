// SPDX-License-Identifier: MIT

#pragma once

#include <neo/algorithm/normalize_energy.hpp>
#include <neo/container/mdspan.hpp>

namespace neo::convolution {

template<inout_object InOutObj>
    requires std::floating_point<value_type_t<InOutObj>>
auto normalize_impulse(InOutObj obj) noexcept -> void
{
    using Index = typename InOutObj::index_type;

    if constexpr (InOutObj::rank() == 1) {
        normalize_energy(obj);
    } else {
        if (obj.extent(0) < 1) {
            return;
        }

        auto channel0 = stdex::submdspan(obj, 0, stdex::full_extent);
        auto factor   = normalize_energy_factor(channel0);
        for (Index ch{1}; ch < obj.extent(0); ++ch) {
            auto channel = stdex::submdspan(obj, ch, stdex::full_extent);
            factor       = std::min(factor, normalize_energy_factor(channel));
        }

        scale(factor, obj);
    }
}

}  // namespace neo::convolution
