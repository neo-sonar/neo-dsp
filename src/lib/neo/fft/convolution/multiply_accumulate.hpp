#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/fill.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/container/sparse_matrix.hpp>

#include <utility>

namespace neo::fft {

template<in_matrix InMatL, in_matrix InMatR, out_vector OutVec>
constexpr auto multiply_accumulate(InMatL lhs, InMatR rhs, OutVec out, std::integral auto shift) -> void
{
    NEO_EXPECTS(lhs.extents() == rhs.extents());
    NEO_EXPECTS(lhs.extent(0) > 0);
    NEO_EXPECTS(lhs.extent(1) > 0);

    // first iteration overwrites output
    auto const l0 = stdex::submdspan(lhs, 0, stdex::full_extent);
    auto const r0 = stdex::submdspan(rhs, shift, stdex::full_extent);
    for (decltype(l0.extent(0)) i{0}; i < l0.extent(0); ++i) {
        out[i] = l0[i] * r0[i];
    }

    // second to n iterations accumulate to output
    auto const num_rows = static_cast<int>(lhs.extent(0));
    for (auto row{1}; row < num_rows; ++row) {
        auto const ridx  = (static_cast<int>(shift) + num_rows - row) % num_rows;
        auto const left  = stdex::submdspan(lhs, row, stdex::full_extent);
        auto const right = stdex::submdspan(rhs, ridx, stdex::full_extent);
        for (decltype(left.extent(0)) i{0}; i < left.extent(0); ++i) {
            out[i] += left[i] * right[i];
        }
    }
}

template<typename U, typename IndexType, typename ValueContainer, typename IndexContainer>
auto multiply_accumulate(
    in_matrix auto lhs,
    sparse_matrix<U, IndexType, ValueContainer, IndexContainer> const& rhs,
    inout_vector auto out,
    std::integral auto shift
) -> void
{
    NEO_EXPECTS(lhs.extent(0) == rhs.rows());
    NEO_EXPECTS(lhs.extent(1) == rhs.columns());
    NEO_EXPECTS(lhs.extent(1) == out.extent(0));
    NEO_EXPECTS(lhs.extent(0) > 0);
    NEO_EXPECTS(lhs.extent(1) > 0);

    auto const& rrows = rhs.row_container();
    auto const& rcols = rhs.column_container();
    auto const& rvals = rhs.value_container();

    // Can't be done on the first iteration, elements are sparse.
    // Won't hit each index like the dense case above
    fill(out, typename decltype(out)::element_type{});

    auto const num_rows = static_cast<int>(lhs.extent(0));
    for (auto row{0}; row < num_rows; ++row) {
        auto const ridx = static_cast<size_t>((static_cast<int>(shift) + num_rows - row) % num_rows);
        auto const left = stdex::submdspan(lhs, row, stdex::full_extent);

        for (auto i{rrows[ridx]}; i < rrows[ridx + 1]; ++i) {
            auto const col = rcols[i];
            out[col] += left[col] * rvals[i];
        }
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
