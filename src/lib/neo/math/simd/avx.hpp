#pragma once

#include <neo/config.hpp>

#include <xmmintrin.h>  // AVX

namespace neo::fft {

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

struct float32x8_t
{
    float32x8_t() = default;

    float32x8_t(__m256 d) : data{d} {}

    __m256 data;
};

struct float64x4_t
{
    float64x4_t() = default;

    float64x4_t(__m256d d) : data{d} {}

    __m256d data;
};

}  // namespace neo::fft
