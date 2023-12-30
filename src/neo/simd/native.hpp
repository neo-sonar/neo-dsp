// SPDX-License-Identifier: MIT
#pragma once

#include <neo/config.hpp>

#if defined(NEO_HAS_SIMD_NEON)
    #include <neo/simd/neon.hpp>
#endif

#if defined(NEO_HAS_SIMD_SSE2)
    #include <neo/simd/sse2.hpp>
#endif

#if defined(NEO_HAS_SIMD_AVX)
    #include <neo/simd/avx.hpp>
#endif

#if defined(NEO_HAS_SIMD_AVX512F)
    #include <neo/simd/avx512.hpp>
#endif
