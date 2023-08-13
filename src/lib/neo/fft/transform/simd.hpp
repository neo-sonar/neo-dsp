#pragma once

#include <neo/config.hpp>

#if defined(NEO_FFT_HAS_SSE2)
    #include <neo/fft/transform/simd/sse2.hpp>
#endif

#if defined(NEO_FFT_HAS_AVX)
    #include <neo/fft/transform/simd/avx.hpp>
#endif

#if defined(NEO_FFT_HAS_AVX512F)
    #include <neo/fft/transform/simd/avx512.hpp>
#endif

namespace neo::fft {

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

#if defined(NEO_FFT_HAS_SSE2)

using complex32x2_t = simd_complex<float32x4_t>;
using complex64x1_t = simd_complex<float64x2_t>;

#endif

#if defined(NEO_FFT_HAS_AVX)

using complex32x4_t = simd_complex<float32x8_t>;
using complex64x2_t = simd_complex<float64x4_t>;

#endif

#if defined(NEO_FFT_HAS_AVX512F)

using complex32x8_t = simd_complex<float32x16_t>;
using complex64x4_t = simd_complex<float64x8_t>;

#endif

}  // namespace neo::fft
