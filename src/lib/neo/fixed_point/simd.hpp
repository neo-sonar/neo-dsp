#pragma once

#include <neo/config.hpp>

#include <neo/fixed_point/fixed_point.hpp>

#if defined(NEO_HAS_SIMD_SSE2)
    #include <neo/simd/sse2.hpp>
#endif

#if defined(NEO_HAS_SIMD_AVX2)
    #include <neo/simd/avx.hpp>
#endif

#if defined(NEO_HAS_SIMD_NEON)
    #include <neo/simd/neon.hpp>
#endif

namespace neo {

#if defined(NEO_HAS_SIMD_SSE2)

struct alignas(16) q7x16
{
    using value_type    = neo::q7;
    using register_type = __m128i;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    q7x16() = default;

    NEO_ALWAYS_INLINE q7x16(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _reg; }

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

    // #if defined(NEO_HAS_SIMD_SSE41)
    // NEO_ALWAYS_INLINE friend auto operator*(q7x16 lhs, q7x16 rhs) -> q7x16
    // {
    //     auto const l = static_cast<register_type>(lhs);
    //     auto const r = static_cast<register_type>(rhs);

    // auto const low_left    = _mm_cvtepi8_epi16(l);
    // auto const low_right   = _mm_cvtepi8_epi16(r);
    // auto const low_product = _mm_mullo_epi16(low_left, low_right);
    // auto const low_shifted = _mm_srli_epi16(low_product, q7::fractional_bits);

    // auto const high_left    = _mm_cvtepi8_epi16(_mm_srli_si128(l, 8));
    // auto const high_right   = _mm_cvtepi8_epi16(_mm_srli_si128(r, 8));
    // auto const high_product = _mm_mullo_epi16(high_left, high_right);
    // auto const high_shifted = _mm_srli_epi16(high_product, q7::fractional_bits);

    // return _mm_packs_epi16(low_shifted, high_shifted);
    // }
    // #endif

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

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _reg; }

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

    // #if defined(NEO_HAS_SIMD_SSE41)
    // NEO_ALWAYS_INLINE friend auto operator*(q15x8 lhs, q15x8 rhs) -> q15x8
    // {
    //     auto const l = static_cast<register_type>(lhs);
    //     auto const r = static_cast<register_type>(rhs);

    // auto const low_left    = _mm_cvtepi16_epi32(l);
    // auto const low_right   = _mm_cvtepi16_epi32(r);
    // auto const low_product = _mm_mullo_epi32(low_left, low_right);
    // auto const low_shifted = _mm_srli_epi32(low_product, q15::fractional_bits);

    // auto const high_left    = _mm_cvtepi16_epi32(_mm_srli_si128(l, 8));
    // auto const high_right   = _mm_cvtepi16_epi32(_mm_srli_si128(r, 8));
    // auto const high_product = _mm_mullo_epi32(high_left, high_right);
    // auto const high_shifted = _mm_srli_epi32(high_product, q15::fractional_bits);

    // return _mm_packs_epi32(low_shifted, high_shifted);
    // }
    // #endif

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

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _reg; }

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

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _reg; }

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

#if defined(NEO_HAS_SIMD_AVX512BW)

struct alignas(64) q7x64
{
    using value_type    = neo::q7;
    using register_type = __m512i;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    q7x64() = default;

    NEO_ALWAYS_INLINE q7x64(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _reg; }

    [[nodiscard]] static auto broadcast(value_type val) -> q7x64 { return _mm512_set1_epi8(val.value()); }

    [[nodiscard]] static auto load_unaligned(value_type const* input) -> q7x64
    {
        auto const* integer = reinterpret_cast<value_type::storage_type const*>(input);
        auto const* ptr     = reinterpret_cast<register_type const*>(integer);
        return _mm512_loadu_si512(ptr);
    }

    auto store_unaligned(value_type* output) const -> void
    {
        auto* integer = reinterpret_cast<value_type::storage_type*>(output);
        auto* ptr     = reinterpret_cast<register_type*>(integer);
        return _mm512_storeu_si512(ptr, _reg);
    }

    NEO_ALWAYS_INLINE friend auto operator+(q7x64 lhs, q7x64 rhs) -> q7x64
    {
        return _mm512_adds_epi8(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(q7x64 lhs, q7x64 rhs) -> q7x64
    {
        return _mm512_subs_epi8(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _reg;
};

struct alignas(64) q15x32
{
    using value_type    = neo::q15;
    using register_type = __m512i;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    q15x32() = default;

    NEO_ALWAYS_INLINE q15x32(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _reg; }

    [[nodiscard]] static auto broadcast(value_type val) -> q15x32 { return _mm512_set1_epi16(val.value()); }

    [[nodiscard]] static auto load_unaligned(value_type const* input) -> q15x32
    {
        auto const* integer = reinterpret_cast<value_type::storage_type const*>(input);
        auto const* ptr     = reinterpret_cast<register_type const*>(integer);
        return _mm512_loadu_si512(ptr);
    }

    auto store_unaligned(value_type* output) const -> void
    {
        auto* integer = reinterpret_cast<value_type::storage_type*>(output);
        auto* ptr     = reinterpret_cast<register_type*>(integer);
        return _mm512_storeu_si512(ptr, _reg);
    }

    NEO_ALWAYS_INLINE friend auto operator+(q15x32 lhs, q15x32 rhs) -> q15x32
    {
        return _mm512_adds_epi16(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(q15x32 lhs, q15x32 rhs) -> q15x32
    {
        return _mm512_subs_epi16(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _reg;
};

#endif

}  // namespace neo
