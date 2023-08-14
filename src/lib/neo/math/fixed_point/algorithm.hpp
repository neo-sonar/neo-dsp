#pragma once

#include <neo/config.hpp>

#include <neo/math/fixed_point/fixed_point.hpp>

#if defined(NEO_HAS_SIMD_SSE2)
    #include <neo/math/simd/sse2.hpp>
#endif

#if defined(NEO_HAS_SIMD_NEON)
    #include <neo/math/simd/neon.hpp>
#endif

#include <functional>
#include <iterator>
#include <span>

namespace neo {

namespace detail {

#if defined(NEO_HAS_SIMD_NEON)
inline constexpr auto const add_kernel_s8  = [](int8x16_t l, int8x16_t r) { return vqaddq_s8(l, r); };
inline constexpr auto const add_kernel_s16 = [](int16x8_t l, int16x8_t r) { return vqaddq_s16(l, r); };
inline constexpr auto const sub_kernel_s8  = [](int8x16_t l, int8x16_t r) { return vqsubq_s8(l, r); };
inline constexpr auto const sub_kernel_s16 = [](int16x8_t l, int16x8_t r) { return vqsubq_s16(l, r); };
inline constexpr auto const mul_kernel_s16 = [](int16x8_t l, int16x8_t r) { return vqdmulhq_s16(l, r); };
#elif defined(NEO_HAS_SIMD_SSE2)
inline constexpr auto const add_kernel_s8  = [](__m128i l, __m128i r) { return _mm_adds_epi8(l, r); };
inline constexpr auto const add_kernel_s16 = [](__m128i l, __m128i r) { return _mm_adds_epi16(l, r); };
inline constexpr auto const sub_kernel_s8  = [](__m128i l, __m128i r) { return _mm_subs_epi8(l, r); };
inline constexpr auto const sub_kernel_s16 = [](__m128i l, __m128i r) { return _mm_subs_epi16(l, r); };
#else
inline constexpr auto const add_kernel_s8  = std::plus<std::int8_t>{};
inline constexpr auto const add_kernel_s16 = std::plus<std::int16_t>{};
inline constexpr auto const sub_kernel_s8  = std::minus<std::int8_t>{};
inline constexpr auto const sub_kernel_s16 = std::minus<std::int16_t>{};
inline constexpr auto const mul_kernel_s8  = std::multiplies<std::int8_t>{};
inline constexpr auto const mul_kernel_s16 = std::multiplies<std::int16_t>{};
#endif

template<int IntegerBits, int FractionalBits, typename StorageType, std::size_t Extent>
auto apply_fixed_point_kernel(
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType> const, Extent> lhs,
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType> const, Extent> rhs,
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType>, Extent> out,
    auto scalar_kernel,
    auto vector_kernel_s8,
    auto vector_kernel_s16
)
{
    NEO_EXPECTS(lhs.size() == rhs.size());
    NEO_EXPECTS(lhs.size() == out.size());

#if defined(NEO_HAS_SIMD_SSE2) || defined(NEO_HAS_SIMD_NEON)
    if constexpr (std::same_as<StorageType, std::int8_t>) {
        simd::apply_kernel<StorageType>(lhs, rhs, out, scalar_kernel, vector_kernel_s8);
        return;
    } else if constexpr (std::same_as<StorageType, std::int16_t>) {
        simd::apply_kernel<StorageType>(lhs, rhs, out, scalar_kernel, vector_kernel_s16);
        return;
    }
#endif

    for (auto i{0U}; i < lhs.size(); ++i) {
        out[i] = scalar_kernel(lhs[i], rhs[i]);
    }
}

}  // namespace detail

/// out[i] = saturate16(lhs[i] + rhs[i])
template<int IntegerBits, int FractionalBits, typename StorageType, std::size_t Extent>
auto add(
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType> const, Extent> lhs,
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType> const, Extent> rhs,
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType>, Extent> out
)
{
    detail::apply_fixed_point_kernel(lhs, rhs, out, std::plus{}, detail::add_kernel_s8, detail::add_kernel_s16);
}

/// out[i] = saturate16(lhs[i] - rhs[i])
template<int IntegerBits, int FractionalBits, typename StorageType, std::size_t Extent>
auto subtract(
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType> const, Extent> lhs,
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType> const, Extent> rhs,
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType>, Extent> out
)
{
    detail::apply_fixed_point_kernel(lhs, rhs, out, std::minus{}, detail::sub_kernel_s8, detail::sub_kernel_s16);
}

/// out[i] = (lhs[i] * rhs[i]) >> FractionalBits;
template<int IntegerBits, int FractionalBits, typename StorageType, std::size_t Extent>
auto multiply(
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType> const, Extent> lhs,
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType> const, Extent> rhs,
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType>, Extent> out
)
{
    NEO_EXPECTS(lhs.size() == rhs.size());
    NEO_EXPECTS(lhs.size() == out.size());

    // Not exactly the same as the other kernels, close enough for now.
    if constexpr (std::same_as<StorageType, std::int16_t> && FractionalBits == 15) {
#if defined(NEO_HAS_SIMD_SSE3)
        simd::apply_kernel<StorageType>(lhs, rhs, out, std::multiplies{}, [](__m128i left, __m128i right) {
            return _mm_mulhrs_epi16(left, right);
        });
        return;
#endif
    }

    if constexpr (std::same_as<StorageType, std::int8_t>) {
#if defined(NEO_HAS_SIMD_SSE41)
        simd::apply_kernel<StorageType>(lhs, rhs, out, std::multiplies{}, [](__m128i left, __m128i right) {
            auto const lowLeft    = _mm_cvtepi8_epi16(left);
            auto const lowRight   = _mm_cvtepi8_epi16(right);
            auto const lowProduct = _mm_mullo_epi16(lowLeft, lowRight);
            auto const lowShifted = _mm_srli_epi16(lowProduct, FractionalBits);

            auto const highLeft    = _mm_cvtepi8_epi16(_mm_srli_si128(left, 8));
            auto const highRight   = _mm_cvtepi8_epi16(_mm_srli_si128(right, 8));
            auto const highProduct = _mm_mullo_epi16(highLeft, highRight);
            auto const highShifted = _mm_srli_epi16(highProduct, FractionalBits);

            return _mm_packs_epi16(lowShifted, highShifted);
        });
        return;
#endif
    } else if constexpr (std::same_as<StorageType, std::int16_t>) {
#if defined(NEO_HAS_SIMD_SSE41)
        simd::apply_kernel<StorageType>(lhs, rhs, out, std::multiplies{}, [](__m128i left, __m128i right) {
            auto const lowLeft    = _mm_cvtepi16_epi32(left);
            auto const lowRight   = _mm_cvtepi16_epi32(right);
            auto const lowProduct = _mm_mullo_epi32(lowLeft, lowRight);
            auto const lowShifted = _mm_srli_epi32(lowProduct, FractionalBits);

            auto const highLeft    = _mm_cvtepi16_epi32(_mm_srli_si128(left, 8));
            auto const highRight   = _mm_cvtepi16_epi32(_mm_srli_si128(right, 8));
            auto const highProduct = _mm_mullo_epi32(highLeft, highRight);
            auto const highShifted = _mm_srli_epi32(highProduct, FractionalBits);

            return _mm_packs_epi32(lowShifted, highShifted);
        });
        return;
#elif defined(NEO_HAS_SIMD_NEON)
        simd::apply_kernel<StorageType>(lhs, rhs, out, std::multiplies{}, detail::mul_kernel_s16);
        return;
#endif
    }

    for (auto i{0U}; i < lhs.size(); ++i) {
        out[i] = std::multiplies{}(lhs[i], rhs[i]);
    }
}

}  // namespace neo
