#pragma once

#include <neo/config.hpp>

#include <xmmintrin.h>  // AVX

namespace neo::simd {

NEO_ALWAYS_INLINE auto cadd(__m256 a, __m256 b) noexcept -> __m256 { return _mm256_add_ps(a, b); }

NEO_ALWAYS_INLINE auto csub(__m256 a, __m256 b) noexcept -> __m256 { return _mm256_sub_ps(a, b); }

NEO_ALWAYS_INLINE auto cmul(__m256 a, __m256 b) noexcept -> __m256
{
    auto real = _mm256_unpacklo_ps(a, b);  // Interleave real parts
    auto imag = _mm256_unpackhi_ps(a, b);  // Interleave imaginary parts
    return _mm256_addsub_ps(_mm256_mul_ps(real, real), _mm256_mul_ps(imag, imag));
}

NEO_ALWAYS_INLINE auto cadd(__m256d a, __m256d b) noexcept -> __m256d { return _mm256_add_pd(a, b); }

NEO_ALWAYS_INLINE auto csub(__m256d a, __m256d b) noexcept -> __m256d { return _mm256_sub_pd(a, b); }

NEO_ALWAYS_INLINE auto cmul(__m256d a, __m256d b) noexcept -> __m256d
{
    auto real = _mm256_unpacklo_pd(a, b);  // Interleave real parts
    auto imag = _mm256_unpackhi_pd(a, b);  // Interleave imaginary parts
    return _mm256_addsub_pd(_mm256_mul_pd(real, real), _mm256_mul_pd(imag, imag));
}

struct float32x8
{
    using register_type = __m256;

    static constexpr auto const alignment = sizeof(register_type);

    float32x8() = default;

    float32x8(register_type val) : _val{val} {}

    [[nodiscard]] operator register_type() const { return _val; }

private:
    register_type _val;
};

struct float64x4
{
    using register_type = __m256d;

    static constexpr auto const alignment = sizeof(register_type);

    float64x4() = default;

    float64x4(register_type val) : _val{val} {}

    [[nodiscard]] operator register_type() const { return _val; }

private:
    register_type _val;
};

}  // namespace neo::simd
