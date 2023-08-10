#pragma once

#include <neo/fft/container/mdspan.hpp>

#include <algorithm>
#include <cstddef>
#include <span>
#include <utility>

namespace neo::fft {

template<typename T>
constexpr auto shift_rows_left(Kokkos::mdspan<T, Kokkos::dextents<size_t, 2>> matrix, std::ptrdiff_t shift) -> void
{
    for (auto ch{0UL}; ch < matrix.extent(0); ++ch) {
        auto channel = std::span{std::addressof(matrix(ch, 0)), matrix.extent(1)};
        std::shift_left(channel.begin(), channel.end(), shift);
    }
}

}  // namespace neo::fft
