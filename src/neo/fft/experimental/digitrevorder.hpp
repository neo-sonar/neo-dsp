// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex/complex.hpp>
#include <neo/container/mdspan.hpp>

namespace neo::fft::experimental {

/// Reorder input using base-radix digit reversal permutation.
template<std::size_t Radix, inout_vector Vec>
    requires(complex<value_type_t<Vec>>)
auto digitrevorder(Vec x) noexcept -> void
{
    auto const len = x.extent(0);

    auto j = 0UL;
    for (auto i = 0UL; i < len - 1UL; i++) {
        if (i < j) {
            std::ranges::swap(x[i], x[j]);
        }
        auto k = (Radix - 1UL) * len / Radix;
        while (k <= j) {
            j -= k;
            k /= Radix;
        }
        j += k / (Radix - 1);
    }
}

}  // namespace neo::fft::experimental
