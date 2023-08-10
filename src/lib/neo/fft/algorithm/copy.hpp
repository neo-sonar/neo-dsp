#pragma once

#include <neo/fft/container/mdspan.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <utility>

namespace neo::fft {

template<in_object InObj, out_object OutObj>
    requires(InObj::rank() == OutObj::rank())
constexpr auto copy(InObj inObj, OutObj outObj) -> void
{
    assert(inObj.extents() == outObj.extents());

    if constexpr (InObj::rank() == 1) {
        for (auto i{0ULL}; i < inObj.extent(0); ++i) { outObj(i) = inObj(i); }
    } else {
        for (auto i{0ULL}; i < inObj.extent(0); ++i) {
            for (auto j{0ULL}; j < inObj.extent(1); ++j) { outObj(i, j) = inObj(i, j); }
        }
    }
}

}  // namespace neo::fft
