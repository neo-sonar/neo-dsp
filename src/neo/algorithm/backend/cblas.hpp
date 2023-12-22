#pragma once

#include <neo/config.hpp>
#include <neo/container/mdspan.hpp>

#include <mkl.h>

#include <complex>
#include <concepts>

namespace neo::cblas {

inline auto scal(std::integral auto n, float alpha, float* x, std::integral auto inc_x) noexcept -> void
{
    cblas_sscal(static_cast<MKL_INT>(n), alpha, x, static_cast<MKL_INT>(inc_x));
}

inline auto scal(std::integral auto n, double alpha, double* x, std::integral auto inc_x) noexcept -> void
{
    cblas_dscal(static_cast<MKL_INT>(n), alpha, x, static_cast<MKL_INT>(inc_x));
}

inline auto
scal(std::integral auto n, std::complex<float> alpha, std::complex<float>* x, std::integral auto inc_x) noexcept -> void
{
    cblas_cscal(static_cast<MKL_INT>(n), &alpha, x, static_cast<MKL_INT>(inc_x));
}

inline auto
scal(std::integral auto n, std::complex<double> alpha, std::complex<double>* x, std::integral auto inc_x) noexcept
    -> void
{
    cblas_zscal(static_cast<MKL_INT>(n), &alpha, x, static_cast<MKL_INT>(inc_x));
}

inline auto scal(std::integral auto n, float alpha, std::complex<float>* x, std::integral auto inc_x) noexcept -> void
{
    cblas_csscal(static_cast<MKL_INT>(n), alpha, x, static_cast<MKL_INT>(inc_x));
}

inline auto scal(std::integral auto n, double alpha, std::complex<double>* x, std::integral auto inc_x) noexcept -> void
{
    cblas_zdscal(static_cast<MKL_INT>(n), alpha, x, static_cast<MKL_INT>(inc_x));
}

}  // namespace neo::cblas
