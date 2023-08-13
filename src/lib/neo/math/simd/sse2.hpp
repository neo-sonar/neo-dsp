#pragma once

#include <neo/config.hpp>

#include <emmintrin.h>
#include <smmintrin.h>  // SSE4

namespace neo::simd {

NEO_ALWAYS_INLINE auto cadd(__m128 a, __m128 b) noexcept -> __m128 { return _mm_add_ps(a, b); }

NEO_ALWAYS_INLINE auto csub(__m128 a, __m128 b) noexcept -> __m128 { return _mm_sub_ps(a, b); }

NEO_ALWAYS_INLINE auto cmul(__m128 a, __m128 b) noexcept -> __m128
{
    auto real = _mm_unpacklo_ps(a, b);  // Interleave real parts
    auto imag = _mm_unpackhi_ps(a, b);  // Interleave imaginary parts
    return _mm_addsub_ps(_mm_mul_ps(real, real), _mm_mul_ps(imag, imag));
}

NEO_ALWAYS_INLINE auto cadd(__m128d a, __m128d b) noexcept -> __m128d { return _mm_add_pd(a, b); }

NEO_ALWAYS_INLINE auto csub(__m128d a, __m128d b) noexcept -> __m128d { return _mm_sub_pd(a, b); }

NEO_ALWAYS_INLINE auto cmul(__m128d a, __m128d b) noexcept -> __m128d
{
    auto real = _mm_unpacklo_pd(a, b);  // Interleave real parts
    auto imag = _mm_unpackhi_pd(a, b);  // Interleave imaginary parts
    return _mm_addsub_pd(_mm_mul_pd(real, real), _mm_mul_pd(imag, imag));
}

struct float32x4_t
{
    using register_type = __m128;

    static constexpr auto const alignment = sizeof(register_type);

    float32x4_t() = default;

    NEO_ALWAYS_INLINE float32x4_t(register_type val) : _val{val} {}

    [[nodiscard]] NEO_ALWAYS_INLINE operator register_type() const { return _val; }

    [[nodiscard]] NEO_ALWAYS_INLINE static auto broadcast(float val) noexcept -> float32x4_t
    {
        return _mm_set1_ps(val);
    }

    auto store_unaligned(float* output) noexcept -> void { return _mm_storeu_ps(output, _val); }

private:
    register_type _val;
};

struct float64x2_t
{
    using register_type = __m128d;

    static constexpr auto const alignment = sizeof(register_type);

    float64x2_t() = default;

    NEO_ALWAYS_INLINE float64x2_t(register_type val) : _val{val} {}

    [[nodiscard]] NEO_ALWAYS_INLINE operator register_type() const { return _val; }

    [[nodiscard]] NEO_ALWAYS_INLINE static auto broadcast(float val) noexcept -> float64x2_t
    {
        return _mm_set1_pd(val);
    }

    auto store_unaligned(double* output) noexcept -> void { return _mm_storeu_pd(output, _val); }

private:
    register_type _val;
};

}  // namespace neo::simd
