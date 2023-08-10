#pragma once

#include <neo/fft/container/mdspan.hpp>

namespace neo::fft {

template<in_matrix InMatL, in_matrix InMatR, out_vector OutVec>
constexpr auto multiply_elementwise_sum_columnwise(InMatL lhs, InMatR rhs, OutVec out, std::size_t shift) -> void
{
    assert(lhs.extents() == rhs.extents());
    assert(lhs.extent(1) > 0);
    assert(shift < lhs.extent(0));

    auto const full = Kokkos::full_extent;

    auto multiply_row = [](in_vector auto left, in_vector auto right, out_vector auto output) -> void {
        for (decltype(left.extent(0)) i{0}; i < left.extent(0); ++i) { output(i) += left(i) * right(i); }
    };

    for (auto row{0U}; row <= shift; ++row) {
        multiply_row(KokkosEx::submdspan(lhs, row, full), KokkosEx::submdspan(rhs, shift - row, full), out);
    }

    for (auto row{shift + 1}; row < lhs.extent(0); ++row) {
        auto const offset    = row - shift;
        auto const offsetRow = lhs.extent(0) - offset;
        multiply_row(KokkosEx::submdspan(lhs, row, full), KokkosEx::submdspan(rhs, offsetRow, full), out);
    }
}

}  // namespace neo::fft
