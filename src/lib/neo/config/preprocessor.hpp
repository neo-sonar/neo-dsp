#pragma once

#define NEO_STRINGIFY(x) #x

#if defined(__GNUC__) || defined(__clang__)
    #define NEO_ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER) && !defined(__clang__)
    #define NEO_ALWAYS_INLINE __forceinline
#else
    #define NEO_ALWAYS_INLINE inline
#endif

#if defined(_MSC_VER)
    #define NEO_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define NEO_RESTRICT __restrict__
#else
    #define NEO_RESTRICT
#endif

#if defined(__ARM_NEON__)
    #define NEO_HAS_SIMD_NEON 1
#endif

#if defined(__SSE2__)
    #define NEO_HAS_SIMD_SSE2 1
#endif

#if defined(__SSE3__)
    #define NEO_HAS_SIMD_SSE3 1
#endif

#if defined(__SSE4_1__)
    #define NEO_HAS_SIMD_SSE41 1
#endif

#if defined(__AVX__)
    #define NEO_HAS_SIMD_AVX 1
#endif

#if defined(__AVX2__)
    #define NEO_HAS_SIMD_AVX2 1
#endif

#if defined(__AVX512F__)
    #define NEO_HAS_SIMD_AVX512F 1
#endif

#if defined(__AVX512BW__)
    #define NEO_HAS_SIMD_AVX512BW 1
#endif
