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

using complex32x2 = complex<float32x4>;
using complex64x1 = complex<float64x2>;

#endif

#if defined(NEO_HAS_AVX)

using complex32x4 = complex<float32x8>;
using complex64x2 = complex<float64x4>;

#endif

#if defined(NEO_HAS_AVX512F)

using complex32x8 = complex<float32x16>;
using complex64x4 = complex<float64x8>;

#endif

}  // namespace neo::simd
