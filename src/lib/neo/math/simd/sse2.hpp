#pragma once

#include <neo/config.hpp>

#include <emmintrin.h>
#include <smmintrin.h>  // SSE4

namespace neo::fft {

NEO_FFT_ALWAYS_INLINE auto cadd(__m128 a, __m128 b) noexcept -> __m128 { return _mm_add_ps(a, b); }

NEO_FFT_ALWAYS_INLINE auto csub(__m128 a, __m128 b) noexcept -> __m128 { return _mm_sub_ps(a, b); }

NEO_FFT_ALWAYS_INLINE auto cmul(__m128 a, __m128 b) noexcept -> __m128
{
    auto real = _mm_unpacklo_ps(a, b);  // Interleave real parts
    auto imag = _mm_unpackhi_ps(a, b);  // Interleave imaginary parts
    return _mm_addsub_ps(_mm_mul_ps(real, real), _mm_mul_ps(imag, imag));
}

NEO_FFT_ALWAYS_INLINE auto cadd(__m128d a, __m128d b) noexcept -> __m128d { return _mm_add_pd(a, b); }

NEO_FFT_ALWAYS_INLINE auto csub(__m128d a, __m128d b) noexcept -> __m128d { return _mm_sub_pd(a, b); }

NEO_FFT_ALWAYS_INLINE auto cmul(__m128d a, __m128d b) noexcept -> __m128d
{
    auto real = _mm_unpacklo_pd(a, b);  // Interleave real parts
    auto imag = _mm_unpackhi_pd(a, b);  // Interleave imaginary parts
    return _mm_addsub_pd(_mm_mul_pd(real, real), _mm_mul_pd(imag, imag));
}

struct float32x4_t
{
    float32x4_t() = default;

    float32x4_t(__m128 d) : data{d} {}

    __m128 data;
};

struct float64x2_t
{
    float64x2_t() = default;

    float64x2_t(__m128d d) : data{d} {}

    __m128d data;
};

}  // namespace neo::fft
