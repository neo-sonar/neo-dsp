#pragma once

#include <neo/config.hpp>

#include <neo/container/mdspan.hpp>
#include <neo/math/complex.hpp>

#include <algorithm>
#include <concepts>

namespace neo {

template<in_object InObj1, in_object InObj2, typename Scalar>
[[nodiscard]] auto allclose(InObj1 lhs, InObj2 rhs, Scalar tolerance) -> bool
{
    NEO_FFT_PRECONDITION(lhs.extents() == rhs.extents());

    if constexpr (InObj1::rank() == 1) {
        for (auto i{0}; std::cmp_less(i, lhs.extent(0)); ++i) {
            if (std::abs(lhs[i] - rhs[i]) > tolerance) {
                return false;
            }
        }
    } else {
        for (auto i{0}; std::cmp_less(i, lhs.extent(0)); ++i) {
            for (auto j{0}; std::cmp_less(j, lhs.extent(1)); ++j) {
                if (std::abs(lhs(i, j) - rhs(i, j)) > tolerance) {
                    return false;
                }
            }
        }
    }

    return true;
}

template<in_object InObj1, in_object InObj2>
[[nodiscard]] auto allclose(InObj1 lhs, InObj2 rhs) -> bool
{
    auto const tolerance = [] {
        using Left  = typename InObj1::value_type;
        using Right = typename InObj2::value_type;
        using Float = decltype(std::abs(std::declval<Left>() - std::declval<Right>()));
        static_assert(std::floating_point<Float>);

        if constexpr (std::same_as<Float, float>) {
            return Float(1e-5);
        }
        return Float(1e-9);
    }();

    return allclose(lhs, rhs, tolerance);
}

}  // namespace neo
