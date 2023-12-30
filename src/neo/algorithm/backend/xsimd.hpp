// SPDX-License-Identifier: MIT
#pragma once

#include <neo/config.hpp>

#if defined(__clang__)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wcast-align"
    #pragma clang diagnostic ignored "-Wdeprecated"
    #pragma clang diagnostic ignored "-Wfloat-conversion"
    #pragma clang diagnostic ignored "-Wfloat-equal"
    #pragma clang diagnostic ignored "-Wshorten-64-to-32"
    #pragma clang diagnostic ignored "-Wsign-conversion"
    #pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#elif defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated"
    #pragma GCC diagnostic ignored "-Wfloat-equal"
    #pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#include <xsimd/xsimd.hpp>

#if defined(__clang__)
    #pragma clang diagnostic pop
#elif defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

#include <concepts>

namespace neo::detail {

template<std::floating_point Float>
    requires(not std::same_as<Float, long double>)
auto multiply_add(
    std::complex<Float> const* x,
    std::complex<Float> const* y,
    std::complex<Float> const* z,
    std::complex<Float>* out,
    std::size_t size
) -> void
{
    using batch_type = xsimd::batch<std::complex<Float>>;

    auto const inc      = batch_type::size;
    auto const vec_size = size - size % inc;

    for (auto i = std::size_t(0); i < vec_size; i += inc) {
        auto const x_vec   = batch_type::load_unaligned(&x[i]);
        auto const y_vec   = batch_type::load_unaligned(&y[i]);
        auto const z_vec   = batch_type::load_unaligned(&z[i]);
        auto const out_vec = x_vec * y_vec + z_vec;
        out_vec.store_unaligned(&out[i]);
    }

    for (auto i = vec_size; i < size; ++i) {
        out[i] = x[i] * y[i] + z[i];
    }
}

template<std::floating_point Float>
    requires(not std::same_as<Float, long double>)
auto multiply_add(
    Float const* x_real,
    Float const* x_imag,
    Float const* y_real,
    Float const* y_imag,
    Float const* z_real,
    Float const* z_imag,
    Float* out_real,
    Float* out_imag,
    std::size_t size
) -> void
{
    using batch_type = xsimd::batch<Float>;

    auto const inc      = batch_type::size;
    auto const vec_size = size - size % inc;

    for (auto i = std::size_t(0); i < vec_size; i += inc) {
        auto const xre = batch_type::load_unaligned(&x_real[i]);
        auto const xim = batch_type::load_unaligned(&x_imag[i]);
        auto const yre = batch_type::load_unaligned(&y_real[i]);
        auto const yim = batch_type::load_unaligned(&y_imag[i]);
        auto const zre = batch_type::load_unaligned(&z_real[i]);
        auto const zim = batch_type::load_unaligned(&z_imag[i]);

        auto const out_re = (xre * yre - xim * yim) + zre;
        auto const out_im = (xre * yim + xim * yre) + zim;

        out_re.store_unaligned(&out_real[i]);
        out_im.store_unaligned(&out_imag[i]);
    }

    for (auto i = vec_size; i < size; ++i) {
        auto const xre = x_real[i];
        auto const xim = x_imag[i];
        auto const yre = y_real[i];
        auto const yim = y_imag[i];

        out_real[i] = (xre * yre - xim * yim) + z_real[i];
        out_imag[i] = (xre * yim + xim * yre) + z_imag[i];
    }
}

}  // namespace neo::detail
