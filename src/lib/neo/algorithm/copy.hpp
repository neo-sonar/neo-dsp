#pragma once

#include <neo/config.hpp>

#include <neo/container/mdspan.hpp>

#include <cassert>

namespace neo {

template<in_object InObj, out_object OutObj>
    requires(InObj::rank() == OutObj::rank())
constexpr auto copy(InObj inObj, OutObj outObj) noexcept -> void
{
    assert(detail::extents_equal(inObj, outObj));

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
