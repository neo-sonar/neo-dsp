#pragma once

#include <neo/config.hpp>
#include <neo/container/mdspan.hpp>

#include <cassert>
#include <utility>

namespace neo::detail {

template<in_object InObj1, in_object InObj2, out_object OutObj, typename Op>
    requires(InObj1::rank() == InObj2::rank() and InObj1::rank() == OutObj::rank())
auto linalg_binary_op(InObj1 x, InObj2 y, OutObj out, Op op) noexcept -> void
{
    assert(detail::extents_equal(x, y, out));

    if constexpr (InObj1::rank() == 1) {
        for (auto i{0}; std::cmp_less(i, x.extent(0)); ++i) {
            out[i] = op(x[i], y[i]);
        }
    } else {
        for (auto i{0}; std::cmp_less(i, x.extent(0)); ++i) {
            for (auto j{0}; std::cmp_less(j, x.extent(1)); ++j) {
                out(i, j) = op(x(i, j), y(i, j));
            }
        }
    }
}

}  // namespace neo::detail
