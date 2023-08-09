#pragma once

#include <neo/fft/config.hpp>

#if defined(__amd64__) or defined(_M_AMD64)
    #include <immintrin.h>  // AVX-512F
    #include <smmintrin.h>  // SSE4
    #include <xmmintrin.h>  // AVX
#endif

namespace neo::fft {

template<typename T>
NEO_FFT_ALWAYS_INLINE auto do_not_optimize(T& value) -> void
{
#if defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#elif defined(__GNUC__)
    asm volatile("" : "+m,r"(value) : : "memory");
#else
    (void)(value);
#endif
}

#if defined(__amd64__) or defined(_M_AMD64)

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

#endif

#ifdef __AVX__

NEO_FFT_ALWAYS_INLINE auto cadd(__m256 a, __m256 b) noexcept -> __m256 { return _mm256_add_ps(a, b); }

NEO_FFT_ALWAYS_INLINE auto csub(__m256 a, __m256 b) noexcept -> __m256 { return _mm256_sub_ps(a, b); }

NEO_FFT_ALWAYS_INLINE auto cmul(__m256 a, __m256 b) noexcept -> __m256
{
    auto real = _mm256_unpacklo_ps(a, b);  // Interleave real parts
    auto imag = _mm256_unpackhi_ps(a, b);  // Interleave imaginary parts
    return _mm256_addsub_ps(_mm256_mul_ps(real, real), _mm256_mul_ps(imag, imag));
}

NEO_FFT_ALWAYS_INLINE auto cadd(__m256d a, __m256d b) noexcept -> __m256d { return _mm256_add_pd(a, b); }

NEO_FFT_ALWAYS_INLINE auto csub(__m256d a, __m256d b) noexcept -> __m256d { return _mm256_sub_pd(a, b); }

NEO_FFT_ALWAYS_INLINE auto cmul(__m256d a, __m256d b) noexcept -> __m256d
{
    auto real = _mm256_unpacklo_pd(a, b);  // Interleave real parts
    auto imag = _mm256_unpackhi_pd(a, b);  // Interleave imaginary parts
    return _mm256_addsub_pd(_mm256_mul_pd(real, real), _mm256_mul_pd(imag, imag));
}
#endif

#ifdef __AVX512F__

NEO_FFT_ALWAYS_INLINE auto cadd(__m512 a, __m512 b) noexcept -> __m512 { return _mm512_add_ps(a, b); }

NEO_FFT_ALWAYS_INLINE auto csub(__m512 a, __m512 b) noexcept -> __m512 { return _mm512_sub_ps(a, b); }

NEO_FFT_ALWAYS_INLINE auto cmul(__m512 a, __m512 b) noexcept -> __m512
{
    // Get real part of a
    __m512 ar = _mm512_shuffle_ps(a, a, _MM_SHUFFLE(2, 2, 0, 0));

    // Get imaginary part of a
    __m512 ai = _mm512_shuffle_ps(a, a, _MM_SHUFFLE(3, 3, 1, 1));

    // Get real part of b
    __m512 br = _mm512_shuffle_ps(b, b, _MM_SHUFFLE(2, 2, 0, 0));

    // Get imaginary part of b
    __m512 bi = _mm512_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 1, 1));

    __m512 real = _mm512_mul_ps(ar, br);  // Multiply real parts
    __m512 imag = _mm512_mul_ps(ai, bi);  // Multiply imaginary parts

    // Subtract and add real and imaginary parts
    __m512 rr = _mm512_sub_ps(real, imag);
    __m512 ri = _mm512_add_ps(_mm512_mul_ps(ar, bi), _mm512_mul_ps(ai, br));

    // Interleave real and imaginary parts
    __m512 result = _mm512_shuffle_ps(rr, ri, _MM_SHUFFLE(2, 0, 2, 0));

    return result;
}

NEO_FFT_ALWAYS_INLINE auto cadd(__m512d a, __m512d b) noexcept -> __m512d { return _mm512_add_pd(a, b); }

NEO_FFT_ALWAYS_INLINE auto csub(__m512d a, __m512d b) noexcept -> __m512d { return _mm512_sub_pd(a, b); }

NEO_FFT_ALWAYS_INLINE auto cmul(__m512d a, __m512d b) noexcept -> __m512d
{
    // Get real part of a
    __m512d ar = _mm512_shuffle_pd(a, a, _MM_SHUFFLE2(0, 0));

    // Get imaginary part of a
    __m512d ai = _mm512_shuffle_pd(a, a, _MM_SHUFFLE2(3, 3));

    // Get real part of b
    __m512d br = _mm512_shuffle_pd(b, b, _MM_SHUFFLE2(0, 0));

    // Get imaginary part of b
    __m512d bi = _mm512_shuffle_pd(b, b, _MM_SHUFFLE2(3, 3));

    __m512d real = _mm512_mul_pd(ar, br);  // Multiply real parts
    __m512d imag = _mm512_mul_pd(ai, bi);  // Multiply imaginary parts

    // Subtract and add real and imaginary parts
    __m512d rr = _mm512_sub_pd(real, imag);
    __m512d ri = _mm512_add_pd(_mm512_mul_pd(ar, bi), _mm512_mul_pd(ai, br));

    // Interleave real and imaginary parts
    __m512d result = _mm512_shuffle_pd(rr, ri, _MM_SHUFFLE2(2, 0));

    return result;
}

#endif

template<typename ValueType>
struct alignas(sizeof(ValueType::data)) simd_complex
{
    simd_complex() noexcept = default;

    simd_complex(decltype(ValueType::data) data) noexcept : _register{data} {}

    NEO_FFT_ALWAYS_INLINE friend auto operator+(simd_complex lhs, simd_complex rhs) noexcept -> simd_complex
    {
        return simd_complex{cadd(lhs._register.data, rhs._register.data)};
    }

    NEO_FFT_ALWAYS_INLINE friend auto operator-(simd_complex lhs, simd_complex rhs) noexcept -> simd_complex
    {
        return simd_complex{csub(lhs._register.data, rhs._register.data)};
    }

    NEO_FFT_ALWAYS_INLINE friend auto operator*(simd_complex lhs, simd_complex rhs) noexcept -> simd_complex
    {
        return simd_complex{cmul(lhs._register.data, rhs._register.data)};
    }

private:
    ValueType _register;
};

#if defined(__amd64__) or defined(_M_AMD64)

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

using complex32x2_t = simd_complex<float32x4_t>;
using complex64x1_t = simd_complex<float64x2_t>;

#endif

#ifdef __AVX__
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

using complex32x4_t = simd_complex<float32x8_t>;
using complex64x2_t = simd_complex<float64x4_t>;

#endif

#ifdef __AVX512F__

struct float32x16_t
{
    float32x16_t() = default;

    float32x16_t(__m512 d) : data{d} {}

    __m512 data;
};

struct float64x8_t
{
    float64x8_t() = default;

    float64x8_t(__m512d d) : data{d} {}

    __m512d data;
};

using complex32x8_t = simd_complex<float32x16_t>;
using complex64x4_t = simd_complex<float64x8_t>;

#endif

}  // namespace neo::fft
