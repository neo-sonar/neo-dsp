#pragma once

#include <neo/config.hpp>

#include <neo/math/complex.hpp>

#if defined(NEO_HAS_SIMD_SSE2)
    #include <neo/math/simd/sse2.hpp>
#endif

#if defined(NEO_HAS_SIMD_AVX)
    #include <neo/math/simd/avx.hpp>
#endif

#if defined(NEO_HAS_SIMD_AVX512F)
    #include <neo/math/simd/avx512.hpp>
#endif

namespace neo::simd {

template<typename FloatBatch>
struct alignas(FloatBatch::alignment) complex
{
    using batch_type       = FloatBatch;
    using register_type    = typename FloatBatch::register_type;
    using real_scalar_type = typename FloatBatch::value_type;

    static constexpr auto const size = FloatBatch::size / 2U;

    complex() noexcept = default;

    complex(FloatBatch batch) noexcept : _batch{batch} {}

    complex(register_type reg) noexcept : _batch{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE operator register_type() const { return static_cast<register_type>(_batch); }

    template<neo::complex Complex>
        requires std::same_as<typename Complex::value_type, real_scalar_type>
    [[nodiscard]] static auto load_unaligned(Complex const* val) -> complex
    {
        return batch_type::load_unaligned(reinterpret_cast<real_scalar_type const*>(val));
    }

    template<neo::complex Complex>
        requires std::same_as<typename Complex::value_type, real_scalar_type>
    auto store_unaligned(Complex* output) const -> void
    {
        return _batch.store_unaligned(reinterpret_cast<real_scalar_type*>(output));
    }

    NEO_ALWAYS_INLINE friend auto operator+(complex lhs, complex rhs) noexcept -> complex
    {
        return complex{cadd(static_cast<register_type>(lhs), static_cast<register_type>(rhs))};
    }

    NEO_ALWAYS_INLINE friend auto operator-(complex lhs, complex rhs) noexcept -> complex
    {
        return complex{csub(static_cast<register_type>(lhs), static_cast<register_type>(rhs))};
    }

    NEO_ALWAYS_INLINE friend auto operator*(complex lhs, complex rhs) noexcept -> complex
    {
        return complex{cmul(static_cast<register_type>(lhs), static_cast<register_type>(rhs))};
    }

private:
    FloatBatch _batch;
};

#if defined(NEO_HAS_SIMD_SSE2)

using complex32x2 = complex<float32x4>;
using complex64x1 = complex<float64x2>;

#endif

#if defined(NEO_HAS_SIMD_AVX)

using complex32x4 = complex<float32x8>;
using complex64x2 = complex<float64x4>;

#endif

#if defined(NEO_HAS_SIMD_AVX512F)

using complex32x8 = complex<float32x16>;
using complex64x4 = complex<float64x8>;

#endif

}  // namespace neo::simd
