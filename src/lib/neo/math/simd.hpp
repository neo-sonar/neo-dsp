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

#include <neo/math/simd/interleave_complex.hpp>
#include <neo/math/simd/parallel_complex.hpp>

namespace neo::simd {

#if defined(NEO_HAS_SIMD_SSE2)
using icomplex64x2  = interleave_complex<float32x4>;
using icomplex128x1 = interleave_complex<float64x2>;
using pcomplex64x4  = parallel_complex<float32x4>;
using pcomplex128x2 = parallel_complex<float64x2>;
#endif

#if defined(NEO_HAS_SIMD_AVX)
using icomplex64x4  = interleave_complex<float32x8>;
using icomplex128x2 = interleave_complex<float64x4>;
using pcomplex64x8  = parallel_complex<float32x8>;
using pcomplex128x4 = parallel_complex<float64x4>;
#endif

#if defined(NEO_HAS_SIMD_AVX512F)
using icomplex64x8  = interleave_complex<float32x16>;
using icomplex128x4 = interleave_complex<float64x8>;
using pcomplex64x16 = parallel_complex<float32x16>;
using pcomplex128x8 = parallel_complex<float64x8>;
#endif

}  // namespace neo::simd
