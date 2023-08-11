#pragma once

#include <neo/fft/container/mdspan.hpp>

#include <algorithm>
#include <complex>
#include <concepts>

namespace neo::fft {

template<in_vector InVec1, in_vector InVec2, typename Scalar>
[[nodiscard]] auto allclose(InVec1 lhs, InVec2 rhs, Scalar tolerance) -> bool
{
    if (lhs.extents() != rhs.extents()) {
        return false;
    }

    for (auto i{0}; std::cmp_less(i, lhs.extent(0)); ++i) {
        if (std::abs(lhs[i] - rhs[i]) > tolerance) {
            return false;
        }
    }

    return true;
}

template<in_vector InVec1, in_vector InVec2>
[[nodiscard]] auto allclose(InVec1 lhs, InVec2 rhs) -> bool
{
    auto const tolerance = [] {
        using Left  = typename InVec1::value_type;
        using Right = typename InVec2::value_type;
        using Float = decltype(std::abs(std::declval<Left>() - std::declval<Right>()));
        static_assert(std::floating_point<Float>);

        if constexpr (std::same_as<Float, float>) {
            return Float(1e-5);
        }
        return Float(1e-9);
    }();

    return allclose(lhs, rhs, tolerance);
}

}  // namespace neo::fft
