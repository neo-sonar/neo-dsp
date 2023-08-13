#pragma once

#include <neo/config.hpp>

#include <neo/container/mdspan.hpp>

namespace neo {

template<in_object InObj, out_object OutObj>
    requires(InObj::rank() == OutObj::rank())
constexpr auto copy(InObj inObj, OutObj outObj) -> void
{
    NEO_EXPECTS(inObj.extents() == outObj.extents());

    if constexpr (InObj::rank() == 1) {
        for (auto i{0ULL}; i < inObj.extent(0); ++i) {
            outObj[i] = inObj[i];
        }
    } else {
        for (auto i{0ULL}; i < inObj.extent(0); ++i) {
            for (auto j{0ULL}; j < inObj.extent(1); ++j) {
                outObj(i, j) = inObj(i, j);
            }
        }
    }
}

}  // namespace neo
