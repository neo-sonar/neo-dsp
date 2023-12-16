#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/detail/linalg_binary_op.hpp>

#include <functional>

namespace neo {

template<in_object InObj1, in_object InObj2, out_object OutObj>
    requires(InObj1::rank() == InObj2::rank() and InObj1::rank() == OutObj::rank())
auto multiply(InObj1 x, InObj2 y, OutObj out) noexcept -> void
{
    return detail::linalg_binary_op(x, y, out, std::multiplies{});
}

}  // namespace neo
