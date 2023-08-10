#pragma once

#include <neo/fft/container/mdspan.hpp>
#include <neo/fft/container/sparse_matrix.hpp>

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

template<typename T, typename U, typename IndexType, typename ValueContainer, typename IndexContainer>
auto multiply_elementwise_sum_columnwise(
    Kokkos::mdspan<T, Kokkos::dextents<std::size_t, 2>> lhs,
    sparse_matrix<U, IndexType, ValueContainer, IndexContainer> const& rhs,
    std::span<U> accumulator,
    std::size_t shift = 0
) -> void
{
    assert(lhs.extent(0) == rhs.rows());
    assert(lhs.extent(1) == rhs.columns());

    auto multiplyRow = [&](auto leftRow, auto rightRow) {
        auto const left   = KokkosEx::submdspan(lhs, leftRow, Kokkos::full_extent);
        auto const& rrows = rhs.row_container();
        auto const& rcols = rhs.column_container();
        auto const& rvals = rhs.value_container();

        for (auto i{rrows[rightRow]}; i < rrows[rightRow + 1]; i++) {
            auto const col = rcols[i];
            accumulator[col] += left(col) * rvals[i];
        }
    };

    for (auto row{0U}; row <= shift; ++row) {
        auto const shifted = shift - row;
        multiplyRow(row, shifted);
    }

    for (auto row{shift + 1}; row < rhs.rows(); ++row) {
        auto const shifted = rhs.rows() - (row - shift);
        multiplyRow(row, shifted);
    }
}

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
auto multiply_elementwise_sum_columnwise(
    sparse_matrix<T, IndexType, ValueContainer, IndexContainer> const& lhs,
    sparse_matrix<T, IndexType, ValueContainer, IndexContainer> const& rhs,
    std::span<T> accumulator
) -> void
{
    assert(lhs.rows() == rhs.rows());
    assert(lhs.columns() == rhs.columns());

    auto const& lrows = lhs.row_container();
    auto const& lcols = lhs.column_container();
    auto const& lvals = lhs.value_container();

    auto const& rrows = rhs.row_container();
    auto const& rcols = rhs.column_container();
    auto const& rvals = rhs.value_container();

    for (auto row(std::size_t{0}); row < lhs.rows(); ++row) {
        auto const rowStart = lrows[row];
        auto const rowEnd   = lrows[row + 1];

        auto const otherRowStart = rrows[row];
        auto const otherRowEnd   = rrows[row + 1];

        auto i = rowStart;
        auto j = otherRowStart;

        while (i < rowEnd && j < otherRowEnd) {
            auto colIndex      = lcols[i];
            auto otherColIndex = rcols[j];

            if (colIndex < otherColIndex) {
                i++;
            } else if (colIndex > otherColIndex) {
                j++;
            } else {
                auto const newValue = lvals[i] * rvals[j];
                accumulator[colIndex] += newValue;
                i++;
                j++;
            }
        }
    }
}

}  // namespace neo::fft
