#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/mean_squared_error.hpp>
#include <neo/container/mdspan.hpp>

#include <cassert>
#include <cmath>
#include <concepts>
#include <type_traits>
#include <utility>

namespace neo {

template<in_vector InVecL, in_vector InVecR>
[[nodiscard]] auto root_mean_squared_error(InVecL lhs, InVecR rhs) noexcept
{
    using std::sqrt;
    return sqrt(mean_squared_error(lhs, rhs));
}

}  // namespace neo
