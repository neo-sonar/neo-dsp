#pragma once

#if defined(__SSE4_1__)
    #include <smmintrin.h>
#endif

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <limits>
#include <span>

namespace neo::fft
{

namespace detail
{

template<typename StorageType>
constexpr auto saturate(std::int32_t x) -> StorageType
{
    auto const min_v = static_cast<std::int32_t>(std::numeric_limits<StorageType>::min());
    auto const max_v = static_cast<std::int32_t>(std::numeric_limits<StorageType>::max());
    return static_cast<StorageType>(std::clamp(x, min_v, max_v));
}

}  // namespace detail

struct underlying_value_t
{
    explicit underlying_value_t() = default;
};

inline constexpr auto underlying_value = underlying_value_t{};

template<int IntegerBits, int FractionalBits, typename StorageType>
struct fixed_point
{
    using storage_type = StorageType;

    static constexpr auto const integer_bits    = IntegerBits;
    static constexpr auto const fractional_bits = FractionalBits;
    static constexpr auto const scale           = static_cast<float>(1 << FractionalBits);
    static constexpr auto const inv_scale       = 1.0F / scale;

    constexpr fixed_point() = default;

    template<std::floating_point Float>
    explicit constexpr fixed_point(Float val) noexcept
        : _value{detail::saturate<storage_type>(static_cast<std::int32_t>(static_cast<float>(val) * scale))}
    {
    }

    constexpr fixed_point([[maybe_unused]] underlying_value_t tag, storage_type val) noexcept : _value{val} {}

    template<std::floating_point Float>
    [[nodiscard]] constexpr operator Float() const noexcept
    {
        return static_cast<float>(_value) * inv_scale;
    }

    [[nodiscard]] constexpr auto value() const noexcept -> storage_type { return _value; }

    [[nodiscard]] constexpr auto operator+() const -> fixed_point { return *this; }

    [[nodiscard]] constexpr auto operator-() const -> fixed_point
    {
        auto const min_v = std::numeric_limits<StorageType>::min();
        auto const max_v = std::numeric_limits<StorageType>::max();

        return fixed_point{
            underlying_value,
            value() == min_v ? max_v : static_cast<StorageType>(-value()),
        };
    }

    friend constexpr auto operator+(fixed_point lhs, fixed_point rhs) -> fixed_point
    {
        return {
            underlying_value,
            detail::saturate<StorageType>(lhs.value() + rhs.value()),
        };
    }

    friend constexpr auto operator-(fixed_point lhs, fixed_point rhs) -> fixed_point
    {
        return {
            underlying_value,
            detail::saturate<StorageType>(lhs.value() - rhs.value()),
        };
    }

    friend constexpr auto operator*(fixed_point lhs, fixed_point rhs) -> fixed_point
    {
        return {
            underlying_value,
            detail::saturate<StorageType>((lhs.value() * rhs.value()) >> fractional_bits),
        };
    }

    friend constexpr auto operator==(fixed_point lhs, fixed_point rhs) -> bool { return lhs.value() == rhs.value(); }
    friend constexpr auto operator!=(fixed_point lhs, fixed_point rhs) -> bool { return lhs.value() != rhs.value(); }
    friend constexpr auto operator<(fixed_point lhs, fixed_point rhs) -> bool { return lhs.value() < rhs.value(); }
    friend constexpr auto operator<=(fixed_point lhs, fixed_point rhs) -> bool { return lhs.value() <= rhs.value(); }
    friend constexpr auto operator>(fixed_point lhs, fixed_point rhs) -> bool { return lhs.value() > rhs.value(); }
    friend constexpr auto operator>=(fixed_point lhs, fixed_point rhs) -> bool { return lhs.value() >= rhs.value(); }

private:
    StorageType _value;
};

using q7_t  = fixed_point<0, 7, std::int8_t>;
using q15_t = fixed_point<0, 15, std::int16_t>;

template<int IntegerBits, int FractionalBits, typename StorageType>
[[nodiscard]] constexpr auto to_float(fixed_point<IntegerBits, FractionalBits, StorageType> val) noexcept -> float
{
    return static_cast<float>(val);
}

template<int IntegerBits, int FractionalBits, typename StorageType>
[[nodiscard]] constexpr auto to_double(fixed_point<IntegerBits, FractionalBits, StorageType> val) noexcept -> double
{
    return static_cast<double>(val);
}

template<int IntegerBits, int FractionalBits, typename StorageType, std::size_t Extent>
auto fixed_point_multiply(std::span<fixed_point<IntegerBits, FractionalBits, StorageType> const, Extent> lhs,
                          std::span<fixed_point<IntegerBits, FractionalBits, StorageType> const, Extent> rhs,
                          std::span<fixed_point<IntegerBits, FractionalBits, StorageType>, Extent> out)
{
    assert(lhs.size() == rhs.size());
    assert(lhs.size() == out.size());

#if defined(__SSE4_1__)
    static constexpr auto vectorSize = 128U / 16U;

    auto sse_kernel = [](__m128i l, __m128i r) -> __m128i
    {
        auto const low_left    = _mm_cvtepi16_epi32(l);
        auto const low_right   = _mm_cvtepi16_epi32(r);
        auto const low_product = _mm_mullo_epi32(low_left, low_right);
        auto const low_result  = _mm_srli_epi32(low_product, FractionalBits);

        auto const high_left    = _mm_cvtepi16_epi32(_mm_srli_si128(l, 8));
        auto const high_right   = _mm_cvtepi16_epi32(_mm_srli_si128(r, 8));
        auto const high_product = _mm_mullo_epi32(high_left, high_right);
        auto const high_result  = _mm_srli_epi32(high_product, FractionalBits);

        return _mm_packs_epi32(low_result, high_result);
    };

    auto const remainder = lhs.size() % vectorSize;
    auto const* lptr     = reinterpret_cast<StorageType const*>(lhs.data());
    auto const* rptr     = reinterpret_cast<StorageType const*>(rhs.data());
    auto const* optr     = reinterpret_cast<StorageType const*>(out.data());

    for (auto i{0U}; i < remainder; ++i) { out[i] = lhs[i] * rhs[i]; }

    for (auto i{remainder}; i < lhs.size(); i += vectorSize)
    {
        auto l = _mm_loadu_si128((__m128i const*)(lptr + i));
        auto r = _mm_loadu_si128((__m128i const*)(rptr + i));
        auto p = sse_kernel(l, r);
        _mm_storeu_si128((__m128i*)(optr + i), p);
    }

#else
    for (auto i{0U}; i < lhs.size(); ++i) { out[i] = lhs[i] * rhs[i]; }
#endif
}

}  // namespace neo::fft
