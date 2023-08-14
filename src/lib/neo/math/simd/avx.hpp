#pragma once

#include <neo/config.hpp>

#include <immintrin.h>

namespace neo::simd {

NEO_ALWAYS_INLINE auto cadd(__m256 a, __m256 b) noexcept -> __m256 { return _mm256_add_ps(a, b); }

NEO_ALWAYS_INLINE auto csub(__m256 a, __m256 b) noexcept -> __m256 { return _mm256_sub_ps(a, b); }

NEO_ALWAYS_INLINE auto cmul(__m256 a, __m256 b) noexcept -> __m256
{
    auto real = _mm256_unpacklo_ps(a, b);  // Interleave real parts
    auto imag = _mm256_unpackhi_ps(a, b);  // Interleave imaginary parts
    return _mm256_addsub_ps(_mm256_mul_ps(real, real), _mm256_mul_ps(imag, imag));
}

NEO_ALWAYS_INLINE auto cadd(__m256d a, __m256d b) noexcept -> __m256d { return _mm256_add_pd(a, b); }

NEO_ALWAYS_INLINE auto csub(__m256d a, __m256d b) noexcept -> __m256d { return _mm256_sub_pd(a, b); }

NEO_ALWAYS_INLINE auto cmul(__m256d a, __m256d b) noexcept -> __m256d
{
    auto real = _mm256_unpacklo_pd(a, b);  // Interleave real parts
    auto imag = _mm256_unpackhi_pd(a, b);  // Interleave imaginary parts
    return _mm256_addsub_pd(_mm256_mul_pd(real, real), _mm256_mul_pd(imag, imag));
}

struct alignas(32) float16x8
{
    using value_type    = _Float16;
    using register_type = __m128i;

    static constexpr auto const alignment = sizeof(register_type);
    static constexpr auto const size      = std::size_t(8);

    float16x8() = default;

    float16x8(register_type val) noexcept : _register{val} {}

    [[nodiscard]] explicit operator register_type() const { return _register; }

    [[nodiscard]] static auto broadcast(value_type val) -> float16x8
    {
        return _mm_set1_epi16(std::bit_cast<std::int16_t>(val));
    }

    [[nodiscard]] static auto load_unaligned(value_type const* input) -> float16x8
    {
        return _mm_loadu_si128(reinterpret_cast<register_type const*>(input));
    }

    auto store_unaligned(value_type* output) const -> void
    {
        return _mm_storeu_si128(reinterpret_cast<register_type*>(output), _register);
    }

    NEO_ALWAYS_INLINE friend auto operator+(float16x8 lhs, float16x8 rhs) -> float16x8
    {
        return binary_op(lhs, rhs, [](auto l, auto r) { return _mm256_add_ps(l, r); });
    }

    NEO_ALWAYS_INLINE friend auto operator-(float16x8 lhs, float16x8 rhs) -> float16x8
    {
        return binary_op(lhs, rhs, [](auto l, auto r) { return _mm256_sub_ps(l, r); });
    }

    NEO_ALWAYS_INLINE friend auto operator*(float16x8 lhs, float16x8 rhs) -> float16x8
    {
        return binary_op(lhs, rhs, [](auto l, auto r) { return _mm256_mul_ps(l, r); });
    }

private:
    NEO_ALWAYS_INLINE friend auto binary_op(float16x8 lhs, float16x8 rhs, auto op) -> float16x8
    {
        auto const left    = _mm256_cvtph_ps(static_cast<register_type>(lhs));
        auto const right   = _mm256_cvtph_ps(static_cast<register_type>(rhs));
        auto const product = op(left, right);
        return _mm256_cvtps_ph(product, _MM_FROUND_CUR_DIRECTION);
    }

    register_type _register;
};

struct alignas(32) float32x8
{
    using value_type    = float;
    using register_type = __m256;

    static constexpr auto const alignment = sizeof(register_type);
    static constexpr auto const size      = std::size_t(8);

    float32x8() = default;

    float32x8(register_type val) noexcept : _val{val} {}

    [[nodiscard]] explicit operator register_type() const { return _val; }

    [[nodiscard]] static auto broadcast(float val) -> float32x8 { return _mm256_set1_ps(val); }

    [[nodiscard]] static auto load_unaligned(float const* input) -> float32x8 { return _mm256_loadu_ps(input); }

    auto store_unaligned(float* output) const -> void { return _mm256_storeu_ps(output, _val); }

    NEO_ALWAYS_INLINE friend auto operator+(float32x8 lhs, float32x8 rhs) -> float32x8
    {
        return _mm256_add_ps(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(float32x8 lhs, float32x8 rhs) -> float32x8
    {
        return _mm256_sub_ps(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator*(float32x8 lhs, float32x8 rhs) -> float32x8
    {
        return _mm256_mul_ps(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _val;
};

struct alignas(32) float64x4
{
    using value_type    = double;
    using register_type = __m256d;

    static constexpr auto const alignment = sizeof(register_type);
    static constexpr auto const size      = std::size_t(4);

    float64x4() = default;

    float64x4(register_type val) noexcept : _val{val} {}

    [[nodiscard]] explicit operator register_type() const { return _val; }

    [[nodiscard]] static auto broadcast(double val) -> float64x4 { return _mm256_set1_pd(val); }

    [[nodiscard]] static auto load_unaligned(double const* input) -> float64x4 { return _mm256_loadu_pd(input); }

    auto store_unaligned(double* output) const -> void { return _mm256_storeu_pd(output, _val); }

    NEO_ALWAYS_INLINE friend auto operator+(float64x4 lhs, float64x4 rhs) -> float64x4
    {
        return _mm256_add_pd(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator-(float64x4 lhs, float64x4 rhs) -> float64x4
    {
        return _mm256_sub_pd(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

    NEO_ALWAYS_INLINE friend auto operator*(float64x4 lhs, float64x4 rhs) -> float64x4
    {
        return _mm256_mul_pd(static_cast<register_type>(lhs), static_cast<register_type>(rhs));
    }

private:
    register_type _val;
};

}  // namespace neo::simd
