#pragma once

#include <neo/config.hpp>

#include <neo/fixed_point/fixed_point.hpp>
#include <neo/fixed_point/simd.hpp>

#include <cassert>
#include <functional>
#include <iterator>
#include <span>

namespace neo {

namespace detail {

template<std::signed_integral IntType, int FractionalBits, std::size_t Extent>
auto apply_fixed_point_kernel(
    std::span<fixed_point<IntType, FractionalBits> const, Extent> lhs,
    std::span<fixed_point<IntType, FractionalBits> const, Extent> rhs,
    std::span<fixed_point<IntType, FractionalBits>, Extent> out,
    auto scalar_kernel,
    auto vector_kernel_s8,
    auto vector_kernel_s16
)
{
    assert(lhs.size() == rhs.size());
    assert(lhs.size() == out.size());

    if constexpr (config::has_simd_sse2 or config::has_simd_neon) {
        if constexpr (std::same_as<IntType, std::int8_t>) {
            simd::apply_kernel<IntType>(lhs, rhs, out, scalar_kernel, vector_kernel_s8);
            return;
        } else if constexpr (std::same_as<IntType, std::int16_t>) {
            simd::apply_kernel<IntType>(lhs, rhs, out, scalar_kernel, vector_kernel_s16);
            return;
        }
    }

    for (auto i{0U}; i < lhs.size(); ++i) {
        out[i] = scalar_kernel(lhs[i], rhs[i]);
    }
}

}  // namespace detail

/// out[i] = saturate16(lhs[i] + rhs[i])
template<std::signed_integral IntType, int FractionalBits, std::size_t Extent>
auto add(
    std::span<fixed_point<IntType, FractionalBits> const, Extent> lhs,
    std::span<fixed_point<IntType, FractionalBits> const, Extent> rhs,
    std::span<fixed_point<IntType, FractionalBits>, Extent> out
)
{
    detail::apply_fixed_point_kernel(lhs, rhs, out, std::plus{}, detail::add_kernel_s8, detail::add_kernel_s16);
}

/// out[i] = saturate16(lhs[i] - rhs[i])
template<std::signed_integral IntType, int FractionalBits, std::size_t Extent>
auto subtract(
    std::span<fixed_point<IntType, FractionalBits> const, Extent> lhs,
    std::span<fixed_point<IntType, FractionalBits> const, Extent> rhs,
    std::span<fixed_point<IntType, FractionalBits>, Extent> out
)
{
    detail::apply_fixed_point_kernel(lhs, rhs, out, std::minus{}, detail::sub_kernel_s8, detail::sub_kernel_s16);
}

/// out[i] = (lhs[i] * rhs[i]) >> FractionalBits;
template<std::signed_integral IntType, int FractionalBits, std::size_t Extent>
auto multiply(
    std::span<fixed_point<IntType, FractionalBits> const, Extent> lhs,
    std::span<fixed_point<IntType, FractionalBits> const, Extent> rhs,
    std::span<fixed_point<IntType, FractionalBits>, Extent> out
)
{
    assert(lhs.size() == rhs.size());
    assert(lhs.size() == out.size());

    if constexpr (std::same_as<IntType, std::int8_t>) {
#if defined(NEO_HAS_SIMD_SSE41)
        simd::apply_kernel<IntType>(lhs, rhs, out, std::multiplies{}, detail::mul_kernel_s8<FractionalBits>);
        return;
#endif
    } else if constexpr (std::same_as<IntType, std::int16_t>) {
#if defined(NEO_HAS_SIMD_SSE41)
        simd::apply_kernel<IntType>(lhs, rhs, out, std::multiplies{}, detail::mul_kernel_s16<FractionalBits>);
        return;
#elif defined(NEO_HAS_SIMD_SSE3)
        // Not exactly the same as the other kernels, close enough for now.
        if constexpr (std::same_as<IntType, std::int16_t> && FractionalBits == 15) {
            simd::apply_kernel<IntType>(lhs, rhs, out, std::multiplies{}, [](__m128i left, __m128i right) {
                return _mm_mulhrs_epi16(left, right);
            });
            return;
        }
#elif defined(NEO_HAS_SIMD_NEON)
        if constexpr (std::same_as<IntType, std::int16_t> && FractionalBits == 15) {
            simd::apply_kernel<IntType>(lhs, rhs, out, std::multiplies{}, detail::mul_kernel_s16);
            return;
        }
#endif
    }

    for (auto i{0U}; i < lhs.size(); ++i) {
        out[i] = std::multiplies{}(lhs[i], rhs[i]);
    }
}

}  // namespace neo
