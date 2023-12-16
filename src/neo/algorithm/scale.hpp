#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/detail/linalg_unary_op.hpp>

namespace neo {

template<typename Scalar, inout_object InOutObj>
constexpr auto scale(Scalar alpha, InOutObj obj) -> void
{
    detail::linalg_unary_op(obj, [alpha](auto const& val) { return val * alpha; });
}

}  // namespace neo
