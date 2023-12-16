#pragma once

#include <neo/config.hpp>

#include <cstddef>
#include <cstdint>
#include <iterator>

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>

namespace neo {

namespace detail {

NEO_ALWAYS_INLINE auto moveldup_ps_sse2(__m128 a) noexcept -> __m128
{
    auto const x = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 2, 2));
    return _mm_shuffle_ps(x, x, _MM_SHUFFLE(0, 0, 2, 2));
}

NEO_ALWAYS_INLINE auto movehdup_ps_sse2(__m128 a) noexcept -> __m128
{
    auto const x = _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 3, 3));
    return _mm_shuffle_ps(x, x, _MM_SHUFFLE(1, 1, 3, 3));
}

NEO_ALWAYS_INLINE auto addsub_sse2(__m128 lhs, __m128 rhs) noexcept -> __m128
{
    auto const mask = _mm_setr_ps(-1.0F, 1.0F, -1.0F, 1.0F);
    return _mm_add_ps(_mm_mul_ps(rhs, mask), lhs);
}

NEO_ALWAYS_INLINE auto cmul_sse2(__m128 lhs, __m128 rhs) noexcept -> __m128
{
    auto const cccc = _mm_mul_ps(lhs, moveldup_ps_sse2(rhs));
    auto const baba = _mm_shuffle_ps(lhs, lhs, 0xB1);
    auto const dddd = _mm_mul_ps(baba, movehdup_ps_sse2(rhs));
    return addsub_sse2(cccc, dddd);
}

}  // namespace detail

NEO_ALWAYS_INLINE auto cadd(__m128 lhs, __m128 rhs) noexcept -> __m128 { return _mm_add_ps(lhs, rhs); }

NEO_ALWAYS_INLINE auto csub(__m128 lhs, __m128 rhs) noexcept -> __m128 { return _mm_sub_ps(lhs, rhs); }

NEO_ALWAYS_INLINE auto cmul(__m128 lhs, __m128 rhs) noexcept -> __m128
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

NEO_ALWAYS_INLINE auto cadd(__m128d lhs, __m128d rhs) noexcept -> __m128d { return _mm_add_pd(lhs, rhs); }

NEO_ALWAYS_INLINE auto csub(__m128d lhs, __m128d rhs) noexcept -> __m128d { return _mm_sub_pd(lhs, rhs); }

struct alignas(16) float32x4
{
    using value_type    = float;
    using register_type = __m128;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    float32x4() = default;

    NEO_ALWAYS_INLINE float32x4(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _reg; }

    [[nodiscard]] NEO_ALWAYS_INLINE static auto broadcast(float reg) noexcept -> float32x4 { return _mm_set1_ps(reg); }

    [[nodiscard]] NEO_ALWAYS_INLINE static auto load_unaligned(float const* input) noexcept -> float32x4
    {
        return _mm_loadu_ps(input);
    }

    NEO_ALWAYS_INLINE auto store_unaligned(float* output) const noexcept -> void { return _mm_storeu_ps(output, _reg); }

    NEO_ALWAYS_INLINE friend auto operator+(float32x4 lhs, float32x4 rhs) noexcept -> float32x4
    {
        return _mm_add_ps(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(float32x4 lhs, float32x4 rhs) noexcept -> float32x4
    {
        return _mm_sub_ps(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator*(float32x4 lhs, float32x4 rhs) noexcept -> float32x4
    {
        return _mm_mul_ps(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _reg;
};

struct alignas(16) float64x2
{
    using value_type    = double;
    using register_type = __m128d;

    static constexpr auto const alignment = alignof(register_type);
    static constexpr auto const size      = sizeof(register_type) / sizeof(value_type);

    float64x2() = default;

    NEO_ALWAYS_INLINE float64x2(register_type reg) noexcept : _reg{reg} {}

    [[nodiscard]] NEO_ALWAYS_INLINE explicit operator register_type() const noexcept { return _reg; }

    [[nodiscard]] NEO_ALWAYS_INLINE static auto broadcast(double reg) noexcept -> float64x2 { return _mm_set1_pd(reg); }

    [[nodiscard]] NEO_ALWAYS_INLINE static auto load_unaligned(double const* input) noexcept -> float64x2
    {
        return _mm_loadu_pd(input);
    }

    NEO_ALWAYS_INLINE auto store_unaligned(double* output) const -> void { return _mm_storeu_pd(output, _reg); }

    NEO_ALWAYS_INLINE friend auto operator+(float64x2 lhs, float64x2 rhs) noexcept -> float64x2
    {
        return _mm_add_pd(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(float64x2 lhs, float64x2 rhs) noexcept -> float64x2
    {
        return _mm_sub_pd(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator*(float64x2 lhs, float64x2 rhs) noexcept -> float64x2
    {
        return _mm_mul_pd(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _reg;
};

namespace simd {

template<typename ScalarType>
inline constexpr auto apply_kernel = [](auto lhs, auto rhs, auto out, auto scalar_kernel, auto vector_kernel) {
    static constexpr auto valueSizeBits = sizeof(ScalarType) * 8UL;
    static constexpr auto vectorSize    = static_cast<ptrdiff_t>(128 / valueSizeBits);
    auto const remainder                = static_cast<ptrdiff_t>(lhs.size()) % vectorSize;

    for (auto i{0}; i < remainder; ++i) {
        out[static_cast<size_t>(i)] = scalar_kernel(lhs[static_cast<size_t>(i)], rhs[static_cast<size_t>(i)]);
    }

    for (auto i{remainder}; i < std::ssize(lhs); i += vectorSize) {
        auto const left  = _mm_loadu_si128(reinterpret_cast<__m128i const*>(std::next(lhs.data(), i)));
        auto const right = _mm_loadu_si128(reinterpret_cast<__m128i const*>(std::next(rhs.data(), i)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(std::next(out.data(), i)), vector_kernel(left, right));
    }
};

}

}  // namespace neo
