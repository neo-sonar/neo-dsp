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

struct float32x8
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

struct float64x4
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
