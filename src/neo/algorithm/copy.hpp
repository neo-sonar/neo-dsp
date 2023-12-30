// SPDX-License-Identifier: MIT
#pragma once

#include <neo/config.hpp>

#include <neo/container/mdspan.hpp>

#include <cassert>

namespace neo {

template<in_object InObj, out_object OutObj>
    requires(InObj::rank() == OutObj::rank())
constexpr auto copy(InObj in_obj, OutObj out_obj) noexcept -> void
{
    assert(detail::extents_equal(in_obj, out_obj));

    if constexpr (InObj::rank() == 1) {
        for (auto i{0ULL}; i < in_obj.extent(0); ++i) {
            out_obj[i] = in_obj[i];
        }
    } else {
        for (auto i{0ULL}; i < in_obj.extent(0); ++i) {
            for (auto j{0ULL}; j < in_obj.extent(1); ++j) {
                out_obj(i, j) = in_obj(i, j);
            }
        }
    }
}

}  // namespace neo
