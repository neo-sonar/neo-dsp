#pragma once

#include <neo/config.hpp>

#include <immintrin.h>

namespace neo::simd {

namespace detail {

NEO_ALWAYS_INLINE auto moveldup_ps_sse2(__m128 a) -> __m128
{
    auto const x = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 0, 0));
    return _mm_shuffle_ps(x, x, _MM_SHUFFLE(0, 0, 0, 0));
}

NEO_ALWAYS_INLINE auto movehdup_ps_sse2(__m128 a) -> __m128
{
    auto const x = _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1));
    return _mm_shuffle_ps(x, x, _MM_SHUFFLE(1, 1, 1, 1));
}

NEO_ALWAYS_INLINE auto addsub_sse2(__m128 lhs, __m128 rhs) -> __m128
{
    auto const mask = _mm_setr_ps(-1.0F, 1.0F, -1.0F, 1.0F);
    return _mm_add_ps(_mm_mul_ps(rhs, mask), lhs);
}

NEO_ALWAYS_INLINE auto cmul_sse2(__m128 lhs, __m128 rhs) -> __m128
{
    auto const cccc = _mm_mul_ps(lhs, moveldup_ps_sse2(rhs));
    auto const baba = _mm_shuffle_ps(lhs, lhs, 0xB1);
    auto const dddd = _mm_mul_ps(baba, movehdup_ps_sse2(rhs));
    return addsub_sse2(cccc, dddd);
}

}  // namespace detail

NEO_ALWAYS_INLINE auto cadd(__m128 lhs, __m128 rhs) -> __m128 { return _mm_add_ps(lhs, rhs); }

NEO_ALWAYS_INLINE auto csub(__m128 lhs, __m128 rhs) -> __m128 { return _mm_sub_ps(lhs, rhs); }

NEO_ALWAYS_INLINE auto cmul(__m128 lhs, __m128 rhs) -> __m128
{
#if defined(NEO_HAS_SIMD_SSE3)
    auto const cccc = _mm_mul_ps(lhs, _mm_moveldup_ps(rhs));
    auto const baba = _mm_shuffle_ps(lhs, lhs, 0xB1);
    auto const dddd = _mm_mul_ps(baba, _mm_movehdup_ps(rhs));
    return _mm_addsub_ps(cccc, dddd);
#else
    return detail::cmul_sse2(lhs, rhs);
#endif
}

NEO_ALWAYS_INLINE auto cadd(__m128d lhs, __m128d rhs) -> __m128d { return _mm_add_pd(lhs, rhs); }

NEO_ALWAYS_INLINE auto csub(__m128d lhs, __m128d rhs) -> __m128d { return _mm_sub_pd(lhs, rhs); }

NEO_ALWAYS_INLINE auto cmul(__m128d lhs, __m128d rhs) -> __m128d
{
    auto real = _mm_unpacklo_pd(lhs, rhs);  // Interleave real parts
    auto imag = _mm_unpackhi_pd(lhs, rhs);  // Interleave imaginary parts
    return _mm_addsub_pd(_mm_mul_pd(real, real), _mm_mul_pd(imag, imag));
}

struct float32x4
{
    using value_type    = float;
    using register_type = __m128;

    static constexpr auto const alignment = sizeof(register_type);
    static constexpr auto const size      = std::size_t(4);

    float32x4() = default;

    NEO_ALWAYS_INLINE float32x4(register_type val) noexcept : _val{val} {}

    [[nodiscard]] NEO_ALWAYS_INLINE operator register_type() const { return _val; }

    [[nodiscard]] static auto broadcast(float val) -> float32x4 { return _mm_set1_ps(val); }

    [[nodiscard]] static auto load_unaligned(float const* input) -> float32x4 { return _mm_loadu_ps(input); }

    auto store_unaligned(float* output) const -> void { return _mm_storeu_ps(output, _val); }

private:
    register_type _val;
};

struct float64x2
{
    using value_type    = double;
    using register_type = __m128d;

    static constexpr auto const alignment = sizeof(register_type);
    static constexpr auto const size      = std::size_t(2);

    float64x2() = default;

    NEO_ALWAYS_INLINE float64x2(register_type val) noexcept : _val{val} {}

    [[nodiscard]] NEO_ALWAYS_INLINE operator register_type() const { return _val; }

    [[nodiscard]] static auto broadcast(double val) -> float64x2 { return _mm_set1_pd(val); }

    [[nodiscard]] static auto load_unaligned(double const* input) -> float64x2 { return _mm_loadu_pd(input); }

    auto store_unaligned(double* output) const -> void { return _mm_storeu_pd(output, _val); }

private:
    register_type _val;
};

}  // namespace neo::simd
