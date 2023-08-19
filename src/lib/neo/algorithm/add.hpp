#pragma once

#include <neo/config.hpp>
#include <neo/container/mdspan.hpp>

#include <cassert>
#include <utility>

namespace neo {

template<in_object InObj1, in_object InObj2, out_object OutObj>
    requires(InObj1::rank() == InObj2::rank() and InObj1::rank() == OutObj::rank())
auto add(InObj1 x, InObj2 y, OutObj out) -> void
{
    assert(x.extents() == y.extents());
    assert(x.extents() == out.extents());

    if constexpr (InObj1::rank() == 1) {
        for (auto i{0}; std::cmp_less(i, x.extent(0)); ++i) {
            out[i] = x[i] + y[i];
        }
    } else {
        for (auto i{0}; std::cmp_less(i, x.extent(0)); ++i) {
            for (auto j{0}; std::cmp_less(j, x.extent(1)); ++j) {
                out(i, j) = x(i, j) + y(i, j);
            }
        }
    }
}

}  // namespace neo
