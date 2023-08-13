#pragma once

#include <neo/config.hpp>

#include <neo/container/mdspan.hpp>
#include <neo/container/sparse_matrix.hpp>

#include <utility>

namespace neo {

template<in_matrix InMatL, in_matrix InMatR, out_vector OutVec>
constexpr auto multiply_sum_columns(InMatL lhs, InMatR rhs, OutVec out) -> void
{
    NEO_EXPECTS(lhs.extents() == rhs.extents());
    NEO_EXPECTS(lhs.extent(1) > 0);

    for (auto row{0}; std::cmp_less(row, lhs.extent(0)); ++row) {
        auto const left  = KokkosEx::submdspan(lhs, row, Kokkos::full_extent);
        auto const right = KokkosEx::submdspan(rhs, row, Kokkos::full_extent);
        for (decltype(left.extent(0)) i{0}; i < left.extent(0); ++i) {
            out[i] += left[i] * right[i];
        }
    }
}

template<typename U, typename IndexType, typename ValueContainer, typename IndexContainer>
auto multiply_sum_columns(
    in_matrix auto lhs,
    sparse_matrix<U, IndexType, ValueContainer, IndexContainer> const& rhs,
    inout_vector auto accumulator
) -> void
{
    NEO_EXPECTS(lhs.extent(0) == rhs.rows());
    NEO_EXPECTS(lhs.extent(1) == rhs.columns());
    NEO_EXPECTS(lhs.extent(1) == accumulator.extent(0));

    for (auto row{0UL}; row < rhs.rows(); ++row) {
        auto const left   = KokkosEx::submdspan(lhs, row, Kokkos::full_extent);
        auto const& rrows = rhs.row_container();
        auto const& rcols = rhs.column_container();
        auto const& rvals = rhs.value_container();

        for (auto i{rrows[row]}; i < rrows[row + 1]; i++) {
            auto const col = rcols[i];
            accumulator[col] += left[col] * rvals[i];
        }
    }
}

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
auto multiply_sum_columns(
    sparse_matrix<T, IndexType, ValueContainer, IndexContainer> const& lhs,
    sparse_matrix<T, IndexType, ValueContainer, IndexContainer> const& rhs,
    std::span<T> accumulator
) -> void
{
    NEO_EXPECTS(lhs.rows() == rhs.rows());
    NEO_EXPECTS(lhs.columns() == rhs.columns());

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

}  // namespace neo
