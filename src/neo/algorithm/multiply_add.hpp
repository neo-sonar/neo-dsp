// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/complex/split_complex.hpp>
#include <neo/container/csr_matrix.hpp>
#include <neo/container/mdspan.hpp>

#if defined(NEO_HAS_APPLE_VDSP)
    #include <Accelerate/Accelerate.h>
#endif

#if defined(NEO_HAS_XSIMD)
    #include <neo/algorithm/backend/xsimd.hpp>
#endif

#include <cassert>
#include <utility>

namespace neo {

// out = x * y + z
template<in_vector VecX, in_vector VecY, in_vector VecZ, out_vector VecOut>
constexpr auto multiply_add(VecX x, VecY y, VecZ z, VecOut out) noexcept -> void
{
    assert(detail::extents_equal(x, y, z, out));

    if constexpr (has_default_accessor<VecX, VecY, VecZ, VecOut> and has_layout_left_or_right<VecX, VecY, VecZ, VecOut>) {
        auto x_ptr   = x.data_handle();
        auto y_ptr   = y.data_handle();
        auto z_ptr   = z.data_handle();
        auto out_ptr = out.data_handle();
        auto size    = x.extent(0);
        if constexpr (requires { detail::multiply_add(x_ptr, y_ptr, z_ptr, out_ptr, size); }) {
            detail::multiply_add(x_ptr, y_ptr, z_ptr, out_ptr, size);
            return;
        }
    }

    for (decltype(x.extent(0)) i{0}; i < x.extent(0); ++i) {
        out[i] = x[i] * y[i] + z[i];
    }
}

// out = x * y + z
template<typename U, typename IndexType, typename ValueContainer, typename IndexContainer>
auto multiply_add(
    in_vector auto x,
    csr_matrix<U, IndexType, ValueContainer, IndexContainer> const& y,
    typename csr_matrix<U, IndexType, ValueContainer, IndexContainer>::index_type y_row,
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

#if defined(NEO_HAS_APPLE_VDSP)
    if constexpr (detail::all_same_value_type_v<VecX, VecY, VecZ, VecOut>) {
        if (detail::strides_equal_to<1>(x.real, x.imag, y.real, y.imag, z.real, z.imag, out.real, out.imag)) {
            using Float = typename VecX::value_type;
            if constexpr (std::same_as<Float, float> or std::same_as<Float, double>) {
                using split_t = std::conditional_t<std::same_as<Float, float>, DSPSplitComplex, DSPDoubleSplitComplex>;
                auto xsc = split_t{.realp = const_cast<Float*>(&x.real[0]), .imagp = const_cast<Float*>(&x.imag[0])};
                auto ysc = split_t{.realp = const_cast<Float*>(&y.real[0]), .imagp = const_cast<Float*>(&y.imag[0])};
                auto zsc = split_t{.realp = const_cast<Float*>(&z.real[0]), .imagp = const_cast<Float*>(&z.imag[0])};
                auto osc = split_t{.realp = &out.real[0], .imagp = &out.imag[0]};

                if constexpr (std::same_as<Float, float>) {
                    vDSP_zvma(&xsc, 1, &ysc, 1, &zsc, 1, &osc, 1, x.real.extent(0));
                } else {
                    vDSP_zvmaD(&xsc, 1, &ysc, 1, &zsc, 1, &osc, 1, x.real.extent(0));
                }

                return;
            }
        }
    }
#endif

    if constexpr (has_default_accessor<VecX, VecY, VecZ, VecOut> and has_layout_left_or_right<VecX, VecY, VecZ, VecOut>) {
        auto const size = static_cast<size_t>(x.real.extent(0));

        auto const* xre = x.real.data_handle();
        auto const* xim = x.imag.data_handle();
        auto const* yre = y.real.data_handle();
        auto const* yim = y.imag.data_handle();
        auto const* zre = z.real.data_handle();
        auto const* zim = z.imag.data_handle();

        auto* ore = out.real.data_handle();
        auto* oim = out.imag.data_handle();

        if constexpr (requires { detail::multiply_add(xre, xim, yre, yim, zre, zim, ore, oim, size); }) {
            detail::multiply_add(xre, xim, yre, yim, zre, zim, ore, oim, size);
            return;
        }
    }

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
