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

}  // namespace neo::detail
