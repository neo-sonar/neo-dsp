#pragma once

#include <neo/config.hpp>

#include <neo/complex/complex.hpp>
#include <neo/simd/native.hpp>

namespace neo {

template<typename FloatBatch>
struct alignas(FloatBatch::alignment) interleave_complex
{
    using batch_type       = FloatBatch;
    using register_type    = typename FloatBatch::register_type;
    using real_scalar_type = typename FloatBatch::value_type;

    static constexpr auto const size = FloatBatch::size / 2U;

    interleave_complex() noexcept = default;

    NEO_ALWAYS_INLINE interleave_complex(FloatBatch batch) noexcept : _batch{batch} {}

    NEO_ALWAYS_INLINE interleave_complex(register_type reg) noexcept : _batch{reg} {}

    template<neo::complex Complex>
        requires std::same_as<typename Complex::value_type, real_scalar_type>
    [[nodiscard]] NEO_ALWAYS_INLINE static auto load_unaligned(Complex const* val) -> interleave_complex
    {
        return batch_type::load_unaligned(reinterpret_cast<real_scalar_type const*>(val));
    }

    template<neo::complex Complex>
        requires std::same_as<typename Complex::value_type, real_scalar_type>
    NEO_ALWAYS_INLINE auto store_unaligned(Complex* output) const -> void
    {
        return _batch.store_unaligned(reinterpret_cast<real_scalar_type*>(output));
    }

    [[nodiscard]] NEO_ALWAYS_INLINE auto batch() const -> batch_type { return _batch; }

    NEO_ALWAYS_INLINE friend auto operator+(interleave_complex lhs, interleave_complex rhs) noexcept
        -> interleave_complex
    {
        return interleave_complex{
            cadd(static_cast<register_type>(lhs.batch()), static_cast<register_type>(rhs.batch())),
        };
    }

    NEO_ALWAYS_INLINE friend auto operator-(interleave_complex lhs, interleave_complex rhs) noexcept
        -> interleave_complex
    {
        return interleave_complex{
            csub(static_cast<register_type>(lhs.batch()), static_cast<register_type>(rhs.batch())),
        };
    }

    NEO_ALWAYS_INLINE friend auto operator*(interleave_complex lhs, interleave_complex rhs) noexcept
        -> interleave_complex
    {
        return interleave_complex{
            cmul(static_cast<register_type>(lhs.batch()), static_cast<register_type>(rhs.batch())),
        };
    }

private:
    FloatBatch _batch;
};

#if defined(NEO_HAS_SIMD_SSE2)
using icomplex64x2  = interleave_complex<float32x4>;
using icomplex128x1 = interleave_complex<float64x2>;
#endif

#if defined(NEO_HAS_SIMD_AVX)
using icomplex64x4  = interleave_complex<float32x8>;
using icomplex128x2 = interleave_complex<float64x4>;
#endif

#if defined(NEO_HAS_BUILTIN_FLOAT16) and defined(NEO_HAS_SIMD_F16C)
using icomplex32x4 = interleave_complex<float16x8>;
using icomplex32x8 = interleave_complex<float16x16>;
#endif

#if defined(NEO_HAS_SIMD_AVX512F)
using icomplex64x8  = interleave_complex<float32x16>;
using icomplex128x4 = interleave_complex<float64x8>;
#endif

template<typename FloatBatch>
inline constexpr auto const is_complex<interleave_complex<FloatBatch>> = true;

}  // namespace neo
