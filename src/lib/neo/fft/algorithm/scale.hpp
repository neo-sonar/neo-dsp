#pragma once

#include <neo/container/mdspan.hpp>

#include <utility>

namespace neo::fft {

template<typename Scalar, inout_object InOutObj>
constexpr auto scale(Scalar alpha, InOutObj obj) -> void
{
    using size_type = typename InOutObj::size_type;

    if constexpr (InOutObj::rank() == 1) {
        for (size_type i{0}; i < obj.extent(0); ++i) {
            obj[i] *= alpha;
        }
    } else {
        for (size_type i{0}; i < obj.extent(0); ++i) {
            for (size_type j{0}; j < obj.extent(1); ++j) {
                obj(i, j) *= alpha;
            }
        }
    }
}

}  // namespace neo::fft
