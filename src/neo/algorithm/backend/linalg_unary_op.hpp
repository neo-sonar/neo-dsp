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
        if constexpr (has_layout_left<InOutObj>) {
            for (index_type col{0}; col < obj.extent(1); ++col) {
                for (index_type row{0}; row < obj.extent(0); ++row) {
                    obj(row, col) = op(obj(row, col));
                }
            }
        } else {
            for (index_type row{0}; row < obj.extent(0); ++row) {
                for (index_type col{0}; col < obj.extent(1); ++col) {
                    obj(row, col) = op(obj(row, col));
                }
            }
        }
    }
}

}  // namespace neo::detail
