#pragma once

#include <neo/config.hpp>

#include <neo/complex/complex.hpp>
#include <neo/simd/native.hpp>

namespace neo {

template<typename FloatBatch>
struct alignas(FloatBatch::alignment) parallel_complex
{
    using batch_type       = FloatBatch;
    using register_type    = typename FloatBatch::register_type;
    using real_scalar_type = typename FloatBatch::value_type;

    static constexpr auto const size = FloatBatch::size;

    parallel_complex() noexcept = default;

    parallel_complex(FloatBatch real, FloatBatch imag) noexcept : _real{real}, _imag{imag} {}

    parallel_complex(register_type real, register_type imag) noexcept : _real{real}, _imag{imag} {}

    [[nodiscard]] NEO_ALWAYS_INLINE auto real() const noexcept -> FloatBatch { return _real; }

    [[nodiscard]] NEO_ALWAYS_INLINE auto imag() const noexcept -> FloatBatch { return _imag; }

    NEO_ALWAYS_INLINE friend auto operator+(parallel_complex lhs, parallel_complex rhs) -> parallel_complex
    {
        return parallel_complex{
            lhs.real() + rhs.real(),
            lhs.imag() + rhs.imag(),
        };
    }

    NEO_ALWAYS_INLINE friend auto operator-(parallel_complex lhs, parallel_complex rhs) -> parallel_complex
    {
        return parallel_complex{
            lhs.real() - rhs.real(),
            lhs.imag() - rhs.imag(),
        };
    }

    NEO_ALWAYS_INLINE friend auto operator*(parallel_complex lhs, parallel_complex rhs) -> parallel_complex
    {
        return parallel_complex{
            lhs.real() * rhs.real() - lhs.imag() * rhs.imag(),
            lhs.real() * rhs.imag() + lhs.imag() * rhs.real(),
        };
    }

private:
    FloatBatch _real;
    FloatBatch _imag;
};

#if defined(NEO_HAS_SIMD_SSE2)
using pcomplex64x4  = parallel_complex<float32x4>;
using pcomplex128x2 = parallel_complex<float64x2>;
#endif

#if defined(NEO_HAS_SIMD_AVX)
using pcomplex64x8  = parallel_complex<float32x8>;
using pcomplex128x4 = parallel_complex<float64x4>;
#endif

#if defined(NEO_HAS_BUILTIN_FLOAT16) and defined(NEO_HAS_SIMD_F16C)
using pcomplex32x8  = parallel_complex<float16x8>;
using pcomplex32x16 = parallel_complex<float16x16>;
#endif

#if defined(NEO_HAS_SIMD_AVX512F)
using pcomplex64x16 = parallel_complex<float32x16>;
using pcomplex128x8 = parallel_complex<float64x8>;
#endif

template<typename FloatBatch>
inline constexpr auto const is_complex<parallel_complex<FloatBatch>> = true;

}  // namespace neo
