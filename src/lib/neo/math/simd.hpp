#pragma once

#include <neo/config.hpp>

#if defined(NEO_HAS_SSE2)
    #include <neo/math/simd/sse2.hpp>
#endif

#if defined(NEO_HAS_AVX)
    #include <neo/math/simd/avx.hpp>
#endif

#if defined(NEO_HAS_AVX512F)
    #include <neo/math/simd/avx512.hpp>
#endif

namespace neo::simd {

template<typename FloatRegister>
struct alignas(FloatRegister::alignment) complex
{
    complex() noexcept = default;

    complex(auto data) noexcept : _register{data} {}

    NEO_ALWAYS_INLINE friend auto operator+(complex lhs, complex rhs) noexcept -> complex
    {
        return complex{cadd(lhs._register, rhs._register)};
    }

    NEO_ALWAYS_INLINE friend auto operator-(complex lhs, complex rhs) noexcept -> complex
    {
        return complex{csub(lhs._register, rhs._register)};
    }

    NEO_ALWAYS_INLINE friend auto operator*(complex lhs, complex rhs) noexcept -> complex
    {
        return complex{cmul(lhs._register, rhs._register)};
    }

private:
    FloatRegister _register;
};

#if defined(NEO_HAS_SSE2)

using complex32x2_t = complex<float32x4_t>;
using complex64x1_t = complex<float64x2_t>;

#endif

#if defined(NEO_HAS_AVX)

using complex32x4_t = complex<float32x8_t>;
using complex64x2_t = complex<float64x4_t>;

#endif

#if defined(NEO_HAS_AVX512F)

using complex32x8_t = complex<float32x16_t>;
using complex64x4_t = complex<float64x8_t>;

#endif

}  // namespace neo::simd
