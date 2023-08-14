#pragma once

#include <neo/config.hpp>

#if defined(NEO_HAS_SIMD_NEON)
    #include <neo/math/simd/neon.hpp>
#endif

#if defined(NEO_HAS_SIMD_SSE2)
    #include <neo/math/simd/sse2.hpp>
#endif

#if defined(NEO_HAS_SIMD_AVX)
    #include <neo/math/simd/avx.hpp>
#endif

#if defined(NEO_HAS_SIMD_AVX512F)
    #include <neo/math/simd/avx512.hpp>
#endif

#include <neo/math/simd/interleaved_complex.hpp>
#include <neo/math/simd/parallel_complex.hpp>

namespace neo::simd {

#if defined(NEO_HAS_SIMD_SSE2)
using icomplex32x2 = interleaved_complex<float32x4>;
using icomplex64x1 = interleaved_complex<float64x2>;
using pcomplex32x4 = parallel_complex<float32x4>;
using pcomplex64x2 = parallel_complex<float64x2>;
#endif

#if defined(NEO_HAS_SIMD_AVX)
using icomplex32x4 = interleaved_complex<float32x8>;
using icomplex64x2 = interleaved_complex<float64x4>;
using pcomplex32x8 = parallel_complex<float32x8>;
using pcomplex64x4 = parallel_complex<float64x4>;
#endif

#if defined(NEO_HAS_SIMD_AVX512F)
using icomplex32x8  = interleaved_complex<float32x16>;
using icomplex64x4  = interleaved_complex<float64x8>;
using pcomplex32x16 = parallel_complex<float32x16>;
using pcomplex64x8  = parallel_complex<float64x8>;
#endif

}  // namespace neo::simd
