#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/fill.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/container/sparse_matrix.hpp>

#include <utility>

namespace neo::fft {

template<in_vector InVecL, in_vector InVecR, inout_vector InOutVec>
constexpr auto multiply_accumulate(InVecL lhs, InVecR rhs, InOutVec out) -> void
{
    NEO_EXPECTS(lhs.extents() == rhs.extents());

    for (decltype(lhs.extent(0)) i{0}; i < lhs.extent(0); ++i) {
        out[i] += lhs[i] * rhs[i];
    }
}

template<typename U, typename IndexType, typename ValueContainer, typename IndexContainer>
auto multiply_accumulate(
    in_vector auto lhs,
    sparse_matrix<U, IndexType, ValueContainer, IndexContainer> const& rhs,
    typename sparse_matrix<U, IndexType, ValueContainer, IndexContainer>::index_type row,
    inout_vector auto out
) -> void
{
    NEO_EXPECTS(lhs.extent(0) == rhs.columns());

    auto const& rrows = rhs.row_container();
    auto const& rcols = rhs.column_container();
    auto const& rvals = rhs.value_container();

    for (auto i{rrows[row]}; i < rrows[row + 1]; ++i) {
        auto const col = rcols[i];
        out[col] += lhs[col] * rvals[i];
    }
}

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
auto multiply_accumulate(
    sparse_matrix<T, IndexType, ValueContainer, IndexContainer> const& lhs,
    sparse_matrix<T, IndexType, ValueContainer, IndexContainer> const& rhs,
    std::span<T> out
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

    fill(out, T{});

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
                ++i;
            } else if (colIndex > otherColIndex) {
                ++j;
            } else {
                auto const newValue = lvals[i] * rvals[j];
                out[colIndex] += newValue;
                ++i;
                ++j;
            }
        }
    }
}

}  // namespace neo::fft
