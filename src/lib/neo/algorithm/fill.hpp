#pragma once

#include <neo/container/mdspan.hpp>

namespace neo {

template<inout_object InOutObj, typename T>
constexpr auto fill(InOutObj obj, T const& val) -> void
{
    if constexpr (InOutObj::rank() == 1) {
        for (auto i{0ULL}; i < obj.extent(0); ++i) {
            obj(i) = val;
        }
    } else {
        for (auto i{0ULL}; i < obj.extent(0); ++i) {
            for (auto j{0ULL}; j < obj.extent(1); ++j) {
                obj(i, j) = val;
            }
        }
    }
}

}  // namespace neo
