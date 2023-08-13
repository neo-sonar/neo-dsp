#pragma once

#if defined(__GNUC__) || defined(__clang__)
    #define NEO_FFT_ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER) && !defined(__clang__)
    #define NEO_FFT_ALWAYS_INLINE __forceinline
#else
    #define NEO_FFT_ALWAYS_INLINE inline
#endif

#if defined(_MSC_VER)
    #define NEO_FFT_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define NEO_FFT_RESTRICT __restrict__
#else
    #define NEO_FFT_RESTRICT
#endif

#define NEO_FFT_STRINGIFY(x) #x

#if defined(__amd64__) or defined(_M_AMD64)
    #define NEO_FFT_HAS_SSE2 1
#endif

#if defined(__AVX__)
    #define NEO_FFT_HAS_AVX 1
#endif

#if defined(__AVX512F__)
    #define NEO_FFT_HAS_AVX512F 1
#endif
