// SPDX-License-Identifier: MIT
#pragma once

#include <neo/algorithm/backend/linalg_unary_op.hpp>

namespace neo {

template<inout_object InOutObj, typename T>
constexpr auto fill(InOutObj obj, T const& val) noexcept -> void
{
    detail::linalg_unary_op(obj, [val](auto const& /*in*/) { return val; });
}

}  // namespace neo
