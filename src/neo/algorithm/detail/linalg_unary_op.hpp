#pragma once

#include <neo/config.hpp>
#include <neo/container/mdspan.hpp>

namespace neo::detail {

template<inout_object InOutObj, typename Op>
constexpr auto linalg_unary_op(InOutObj obj, Op op) noexcept -> void
{
    using index_type = typename InOutObj::index_type;

    if constexpr (InOutObj::rank() == 1) {
        for (index_type i{0}; i < obj.extent(0); ++i) {
            obj(i) = op(obj(i));
        }
    } else {
        for (index_type i{0}; i < obj.extent(0); ++i) {
            for (index_type j{0}; j < obj.extent(1); ++j) {
                obj(i, j) = op(obj(i, j));
            }
        }
    }
}

}  // namespace neo::detail
