#pragma once

#include <neo/fft/config.hpp>

#include <neo/fft/container/mdspan.hpp>

#include <cmath>
#include <concepts>
#include <type_traits>
#include <utility>

namespace neo::fft {

template<in_vector InVecL, in_vector InVecR>
auto rms_error(InVecL lhs, InVecR rhs)
{
    NEO_FFT_PRECONDITION(lhs.extents() == rhs.extents());

    using LeftReal  = typename InVecL::value_type;
    using RightReal = typename InVecR::value_type;
    using Float     = decltype(std::declval<LeftReal>() - std::declval<RightReal>());
    using Index     = std::common_type_t<typename InVecL::index_type, typename InVecR::index_type>;

    auto sum = Float(0);
    for (Index i{0}; std::cmp_less(i, lhs.extent(0)); ++i) {
        auto const diff    = lhs[i] - rhs[i];
        auto const squared = diff * diff;
        sum += squared;
    }

    using std::sqrt;
    return sqrt(sum / static_cast<Float>(lhs.extent(0)));
}

}  // namespace neo::fft
