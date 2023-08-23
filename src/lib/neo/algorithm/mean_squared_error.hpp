#pragma once

#include <neo/config.hpp>

#include <neo/complex/complex.hpp>
#include <neo/container/mdspan.hpp>

#include <cassert>
#include <concepts>

namespace neo {

template<in_vector InVecL, in_vector InVecR>
[[nodiscard]] auto mean_squared_error(InVecL lhs, InVecR rhs) noexcept
{
    using Index = std::common_type_t<typename InVecL::index_type, typename InVecR::index_type>;
    using Float = std::common_type_t<typename InVecL::value_type, typename InVecR::value_type>;

    static_assert(std::floating_point<Float>);
    assert(lhs.extents() == rhs.extents());

    auto abs_if_needed = [](auto val) {
        if constexpr (complex<Float>) {
            using std::abs;
            return abs(val);
        } else {
            return val;
        }
    };

    auto sum = Float(0);
    for (Index i{0}; std::cmp_less(i, lhs.extent(0)); ++i) {
        auto const diff    = abs_if_needed(lhs[i]) - abs_if_needed(rhs[i]);
        auto const squared = diff * diff;
        sum += squared;
    }

    return sum / static_cast<Float>(lhs.size());
}

}  // namespace neo
