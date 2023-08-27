#pragma once

#include <neo/config.hpp>

#include <neo/container/mdspan.hpp>

#include <cassert>

namespace neo {

template<in_vector Vec>
struct split_complex
{
    Vec real;
    Vec imag;
};

template<in_vector Vec>
split_complex(Vec re, Vec im) -> split_complex<Vec>;

template<in_vector VecX, in_vector VecY, out_vector VecOut>
constexpr auto add(split_complex<VecX> x, split_complex<VecY> y, split_complex<VecOut> out) noexcept -> void
{
    assert(detail::extents_equal(x.real, x.imag, y.real, y.imag, out.real, out.imag));

    for (auto i{0}; i < static_cast<int>(x.real.extent(0)); ++i) {
        out.real[i] = x.real[i] + y.real[i];
        out.imag[i] = x.imag[i] + y.imag[i];
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
