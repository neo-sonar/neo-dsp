#pragma once

#include <algorithm>
#include <complex>
#include <concepts>
#include <functional>
#include <numeric>
#include <span>

namespace neo::fft {

[[nodiscard]] auto allclose(std::span<float const> lhs, std::span<float const> rhs, float tolerance = 1e-5F) -> bool;

[[nodiscard]] auto allclose(std::span<double const> lhs, std::span<double const> rhs, double tolerance = 1e-9) -> bool;

[[nodiscard]] auto
allclose(std::span<std::complex<float> const> lhs, std::span<std::complex<float> const> rhs, float tolerance = 1e-5F)
    -> bool;

[[nodiscard]] auto
allclose(std::span<std::complex<double> const> lhs, std::span<std::complex<double> const> rhs, double tolerance = 1e-9)
    -> bool;

namespace detail {

template<typename T, typename U = T>
auto allclose_impl(std::span<T const> lhs, std::span<T const> rhs, U tolerance) -> bool
{
    if (lhs.size() != rhs.size()) { return false; }
    return std::transform_reduce(
        lhs.begin(),
        lhs.end(),
        rhs.begin(),
        true,
        std::logical_and{},
        [tolerance](auto l, auto r) { return std::abs(l - r) < tolerance; }
    );
}

}  // namespace detail

inline auto allclose(std::span<float const> lhs, std::span<float const> rhs, float tolerance) -> bool
{
    return detail::allclose_impl<float>(lhs, rhs, tolerance);
}

inline auto allclose(std::span<double const> lhs, std::span<double const> rhs, double tolerance) -> bool
{
    return detail::allclose_impl<double>(lhs, rhs, tolerance);
}

inline auto
allclose(std::span<std::complex<float> const> lhs, std::span<std::complex<float> const> rhs, float tolerance) -> bool
{
    return detail::allclose_impl<std::complex<float>, float>(lhs, rhs, tolerance);
}

inline auto
allclose(std::span<std::complex<double> const> lhs, std::span<std::complex<double> const> rhs, double tolerance) -> bool
{
    return detail::allclose_impl<std::complex<double>, double>(lhs, rhs, tolerance);
}

}  // namespace neo::fft
