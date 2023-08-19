#pragma once

#define NEO_STRINGIFY(x) #x

#if defined(_MSC_VER) && !defined(__clang__)
    #define NEO_COMPILER_MSVC 1
#elif defined(__GNUC__) && !defined(__clang__)
    #define NEO_COMPILER_GCC 1
#elif defined(__clang__)
    #define NEO_COMPILER_CLANG 1
#else
    #define NEO_COMPILER_UNKOWN 1
#endif

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

#if defined(__F16C__)
    #define NEO_HAS_SIMD_F16C 1
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

#if defined(_WIN32)
    #define NEO_PLATFORM_WINDOWS 1
#endif

#if defined(__APPLE__)
    #include <TargetConditionals.h>
    #define NEO_PLATFORM_APPLE 1
    #if TARGET_OS_MAC == 1
        #define NEO_PLATFORM_MACOS
    #else
        #error "Unsupported Apple platform!"
    #endif
#endif

#if defined(__linux__) and not defined(__ANDROID__)
    #define NEO_PLATFORM_LINUX 1
#endif

#if defined(__ANDROID__)
    #define NEO_PLATFORM_ANDROID 1
#endif

#if defined(__FreeBSD__)
    #define NEO_PLATFORM_FREEBSD 1
#endif

#if defined(__OpenBSD__)
    #define NEO_PLATFORM_OPENBSD 1
#endif

#if defined(NEO_PLATFORM_MACOS)
    #define NEO_HAS_BUILTIN_FLOAT16 1
#elif defined(NEO_HAS_SIMD_SSE41) and not defined(NEO_PLATFORM_LINUX)
    #define NEO_HAS_BUILTIN_FLOAT16 1
#endif

namespace neo {

template<typename>
inline constexpr auto const always_false = false;

}  // namespace neo
