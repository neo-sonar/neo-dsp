#pragma once

#include <neo/config.hpp>

#include <neo/complex/split_complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/container/sparse_matrix.hpp>

#include <cassert>
#include <utility>

namespace neo {

// out = x * y + z
template<in_vector InVecX, in_vector InVecY, in_vector InVecZ, out_vector OutVec>
constexpr auto multiply_add(InVecX x, InVecY y, InVecZ z, OutVec out) noexcept -> void
{
    assert(detail::extents_equal(x, y, z, out));

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

// out = x * y + z
template<in_vector VecX, in_vector VecY, in_vector VecZ, out_vector VecOut>
constexpr auto
multiply_add(split_complex<VecX> x, split_complex<VecY> y, split_complex<VecZ> z, split_complex<VecOut> out) noexcept
    -> void
{
    assert(detail::extents_equal(x.real, x.imag, y.real, y.imag, z.real, z.imag, out.real, out.imag));

    for (auto i{0}; i < static_cast<int>(x.real.extent(0)); ++i) {
        auto const xre = x.real[i];
        auto const xim = x.imag[i];
        auto const yre = y.real[i];
        auto const yim = y.imag[i];

        out.real[i] = (xre * yre - xim * yim) + z.real[i];
        out.imag[i] = (xre * yim + xim * yre) + z.imag[i];
    }
}

}  // namespace neo
