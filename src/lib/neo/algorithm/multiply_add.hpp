#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/fill.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/container/sparse_matrix.hpp>

#include <cassert>
#include <utility>

namespace neo {

// out = x * y + z
template<in_vector InVecX, in_vector InVecY, in_vector InVecZ, out_vector OutVec>
constexpr auto multiply_add(InVecX x, InVecY y, InVecZ z, OutVec out) noexcept -> void
{
    assert(x.extents() == y.extents());
    assert(x.extents() == z.extents());
    assert(x.extents() == out.extents());

    for (decltype(x.extent(0)) i{0}; i < x.extent(0); ++i) {
        out[i] = x[i] * y[i] + z[i];
    }
}

// out = x * y + z
template<typename U, typename IndexType, typename ValueContainer, typename IndexContainer>
auto multiply_add(
    in_vector auto x,
    sparse_matrix<U, IndexType, ValueContainer, IndexContainer> const& y,
    typename sparse_matrix<U, IndexType, ValueContainer, IndexContainer>::index_type y_row,
    in_vector auto z,
    out_vector auto out
) noexcept -> void
{
    assert(x.extent(0) == y.columns());

    auto const& rrows = y.row_container();
    auto const& rcols = y.column_container();
    auto const& rvals = y.value_container();

    for (auto i{rrows[y_row]}; i < rrows[y_row + 1]; ++i) {
        auto col = rcols[i];
        out[col] = x[col] * rvals[i] + z[col];
    }
}

}  // namespace neo
