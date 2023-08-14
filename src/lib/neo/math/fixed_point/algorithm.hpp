#pragma once

#include <neo/config.hpp>

#include <neo/math/fixed_point/fixed_point.hpp>

#if defined(NEO_HAS_SIMD_SSE2)
    #include <neo/math/simd/sse2.hpp>
#endif

#if defined(NEO_HAS_SIMD_AVX2)
    #include <neo/math/simd/avx.hpp>
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
inline constexpr auto const add_kernel_s8  = std::plus<q7>{};
inline constexpr auto const add_kernel_s16 = std::plus<q15>{};
inline constexpr auto const sub_kernel_s8  = std::minus<q7>{};
inline constexpr auto const sub_kernel_s16 = std::minus<q15>{};
inline constexpr auto const mul_kernel_s8  = std::multiplies<q7>{};
inline constexpr auto const mul_kernel_s16 = std::multiplies<q15>{};
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

#elif defined(NEO_HAS_SIMD_SSE3)
        // Not exactly the same as the other kernels, close enough for now.
        if constexpr (std::same_as<StorageType, std::int16_t> && FractionalBits == 15) {
            simd::apply_kernel<StorageType>(lhs, rhs, out, std::multiplies{}, [](__m128i left, __m128i right) {
                return _mm_mulhrs_epi16(left, right);
            });
            return;
        }
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

namespace neo::simd {

#if defined(NEO_HAS_SIMD_SSE2)

struct alignas(16) q7x16
{
    using value_type    = neo::q7;
    using register_type = __m128i;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    q7x16() = default;

    NEO_ALWAYS_INLINE q7x16(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const { return _reg; }

    [[nodiscard]] static auto broadcast(value_type val) -> q7x16 { return _mm_set1_epi8(val.value()); }

    [[nodiscard]] static auto load_unaligned(value_type const* input) -> q7x16
    {
        auto const* integer = reinterpret_cast<value_type::storage_type const*>(input);
        auto const* ptr     = reinterpret_cast<register_type const*>(integer);
        return _mm_loadu_si128(ptr);
    }

    auto store_unaligned(value_type* output) const -> void
    {
        auto* integer = reinterpret_cast<value_type::storage_type*>(output);
        auto* ptr     = reinterpret_cast<register_type*>(integer);
        return _mm_storeu_si128(ptr, _reg);
    }

    NEO_ALWAYS_INLINE friend auto operator+(q7x16 lhs, q7x16 rhs) -> q7x16
    {
        return _mm_adds_epi8(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(q7x16 lhs, q7x16 rhs) -> q7x16
    {
        return _mm_subs_epi8(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _reg;
};

struct alignas(16) q15x8
{
    using value_type    = neo::q15;
    using register_type = __m128i;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    q15x8() = default;

    NEO_ALWAYS_INLINE q15x8(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const { return _reg; }

    [[nodiscard]] static auto broadcast(value_type val) -> q15x8 { return _mm_set1_epi16(val.value()); }

    [[nodiscard]] static auto load_unaligned(value_type const* input) -> q15x8
    {
        auto const* integer = reinterpret_cast<value_type::storage_type const*>(input);
        auto const* ptr     = reinterpret_cast<register_type const*>(integer);
        return _mm_loadu_si128(ptr);
    }

    auto store_unaligned(value_type* output) const -> void
    {
        auto* integer = reinterpret_cast<value_type::storage_type*>(output);
        auto* ptr     = reinterpret_cast<register_type*>(integer);
        return _mm_storeu_si128(ptr, _reg);
    }

    NEO_ALWAYS_INLINE friend auto operator+(q15x8 lhs, q15x8 rhs) -> q15x8
    {
        return _mm_adds_epi16(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(q15x8 lhs, q15x8 rhs) -> q15x8
    {
        return _mm_subs_epi16(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _reg;
};

#endif

#if defined(NEO_HAS_SIMD_AVX2)

struct alignas(32) q7x32
{
    using value_type    = neo::q7;
    using register_type = __m256i;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    q7x32() = default;

    NEO_ALWAYS_INLINE q7x32(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const { return _reg; }

    [[nodiscard]] static auto broadcast(value_type val) -> q7x32 { return _mm256_set1_epi8(val.value()); }

    [[nodiscard]] static auto load_unaligned(value_type const* input) -> q7x32
    {
        auto const* integer = reinterpret_cast<value_type::storage_type const*>(input);
        auto const* ptr     = reinterpret_cast<register_type const*>(integer);
        return _mm256_loadu_si256(ptr);
    }

    auto store_unaligned(value_type* output) const -> void
    {
        auto* integer = reinterpret_cast<value_type::storage_type*>(output);
        auto* ptr     = reinterpret_cast<register_type*>(integer);
        return _mm256_storeu_si256(ptr, _reg);
    }

    NEO_ALWAYS_INLINE friend auto operator+(q7x32 lhs, q7x32 rhs) -> q7x32
    {
        return _mm256_adds_epi8(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(q7x32 lhs, q7x32 rhs) -> q7x32
    {
        return _mm256_subs_epi8(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _reg;
};

struct alignas(32) q15x16
{
    using value_type    = neo::q15;
    using register_type = __m256i;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    q15x16() = default;

    NEO_ALWAYS_INLINE q15x16(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const { return _reg; }

    [[nodiscard]] static auto broadcast(value_type val) -> q15x16 { return _mm256_set1_epi16(val.value()); }

    [[nodiscard]] static auto load_unaligned(value_type const* input) -> q15x16
    {
        auto const* integer = reinterpret_cast<value_type::storage_type const*>(input);
        auto const* ptr     = reinterpret_cast<register_type const*>(integer);
        return _mm256_loadu_si256(ptr);
    }

    auto store_unaligned(value_type* output) const -> void
    {
        auto* integer = reinterpret_cast<value_type::storage_type*>(output);
        auto* ptr     = reinterpret_cast<register_type*>(integer);
        return _mm256_storeu_si256(ptr, _reg);
    }

    NEO_ALWAYS_INLINE friend auto operator+(q15x16 lhs, q15x16 rhs) -> q15x16
    {
        return _mm256_adds_epi16(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(q15x16 lhs, q15x16 rhs) -> q15x16
    {
        return _mm256_subs_epi16(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _reg;
};

#endif

}  // namespace neo::simd
