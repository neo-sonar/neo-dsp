#pragma once

#include <neo/config.hpp>
#include <neo/container/mdspan.hpp>

#include <mkl.h>

#include <complex>

namespace neo::detail {

template<typename T>
inline constexpr auto const is_blas_type = false;

template<std::floating_point Float>
inline constexpr auto const is_blas_type<Float> = true;

template<std::floating_point Float>
inline constexpr auto const is_blas_type<std::complex<Float>> = true;

template<typename>
struct cblas_traits;

template<>
struct cblas_traits<float>
{
    static constexpr auto scale = cblas_sscal;
};

template<>
struct cblas_traits<double>
{
    static constexpr auto scale = cblas_dscal;
};

template<>
struct cblas_traits<std::complex<float>>
{
    static constexpr auto scale = cblas_cscal;
};

template<>
struct cblas_traits<std::complex<double>>
{
    static constexpr auto scale = cblas_zscal;
};

}  // namespace neo::detail
