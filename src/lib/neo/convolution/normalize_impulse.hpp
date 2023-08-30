#pragma once

#include <neo/algorithm/energy_normalize.hpp>
#include <neo/container/mdspan.hpp>

namespace neo {

template<inout_object InOutObj>
    requires std::floating_point<typename InOutObj::value_type>
auto normalize_impulse(InOutObj obj) noexcept -> void
{
    using Index = typename InOutObj::index_type;

    if constexpr (InOutObj::rank() == 1) {
        energy_normalize(obj);
    } else {
        if (obj.extent(0) < 1) {
            return;
        }

        auto channel0 = stdex::submdspan(obj, 0, stdex::full_extent);
        auto factor   = energy_normalization_factor(channel0);
        for (Index ch{1}; ch < obj.extent(0); ++ch) {
            auto channel = stdex::submdspan(obj, ch, stdex::full_extent);
            factor       = (std::min)(factor, energy_normalization_factor(channel));
        }

        scale(factor, obj);
    }
}

}  // namespace neo
