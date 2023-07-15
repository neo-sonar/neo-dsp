#pragma once

#if defined(__SSE2__)
    #include <smmintrin.h>
#endif

#if defined(__SSE3__)
    #include <tmmintrin.h>
#endif

#if defined(__ARM_NEON__)
    #include <arm_neon.h>
#endif

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <type_traits>

namespace neo::fft {

namespace detail {

template<typename... T>
constexpr bool always_false = false;

template<typename StorageType>
constexpr auto saturate(std::int32_t x) -> StorageType
{
    auto const min_v = static_cast<std::int32_t>(std::numeric_limits<StorageType>::min());
    auto const max_v = static_cast<std::int32_t>(std::numeric_limits<StorageType>::max());
    return static_cast<StorageType>(std::clamp(x, min_v, max_v));
}

#if defined(__SSE2__)
template<int ValueSizeBits>
inline constexpr auto apply_fixed_point_kernel_sse
    = [](auto const& lhs, auto const& rhs, auto const& out, auto scalar_kernel, auto vector_kernel) {
    static constexpr auto vectorSize = static_cast<ptrdiff_t>(128 / ValueSizeBits);
    auto const remainder             = static_cast<ptrdiff_t>(lhs.size()) % vectorSize;

    for (auto i{0}; i < remainder; ++i) {
        out[static_cast<size_t>(i)] = scalar_kernel(lhs[static_cast<size_t>(i)], rhs[static_cast<size_t>(i)]);
    }

    for (auto i{remainder}; i < lhs.size(); i += vectorSize) {
        auto const left  = _mm_loadu_si128(reinterpret_cast<__m128i const*>(std::next(lhs.data(), i)));
        auto const right = _mm_loadu_si128(reinterpret_cast<__m128i const*>(std::next(rhs.data(), i)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(std::next(out.data(), i)), vector_kernel(left, right));
    }
};
#endif

#if defined(__ARM_NEON__)
template<int ValueSizeBits>
inline constexpr auto apply_fixed_point_kernel_neon128
    = [](auto const& lhs, auto const& rhs, auto const& out, auto scalar_kernel, auto vector_kernel) {

    auto load = [](auto* ptr) {
        if constexpr (ValueSizeBits == 8) {
            return vld1q_s8(reinterpret_cast<std::int8_t const*>(ptr));
        } else {
            static_assert(ValueSizeBits == 16);
            return vld1q_s16(reinterpret_cast<std::int16_t const*>(ptr));
        }
    };

    auto store = [](auto* ptr, auto val) {
        if constexpr (ValueSizeBits == 8) {
            return vst1q_s8(reinterpret_cast<std::int8_t*>(ptr), val);
        } else {
            static_assert(ValueSizeBits == 16);
            return vst1q_s16(reinterpret_cast<std::int16_t*>(ptr), val);
        }
    };

    static constexpr auto vectorSize = static_cast<ptrdiff_t>(128 / ValueSizeBits);
    auto const remainder             = static_cast<ptrdiff_t>(lhs.size()) % vectorSize;

    for (auto i{0}; i < remainder; ++i) {
        out[static_cast<size_t>(i)] = scalar_kernel(lhs[static_cast<size_t>(i)], rhs[static_cast<size_t>(i)]);
    }

    for (auto i{remainder}; i < lhs.size(); i += vectorSize) {
        auto const left  = load(std::next(lhs.data(), i));
        auto const right = load(std::next(rhs.data(), i));
        store(std::next(out.data(), i), vector_kernel(left, right));
    }
};
#endif

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
    {}

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

/// out[i] = saturate16(lhs[i] + rhs[i])
template<int IntegerBits, int FractionalBits, typename StorageType, std::size_t Extent>
auto add(
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType> const, Extent> lhs,
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType> const, Extent> rhs,
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType>, Extent> out
)
{
    assert(lhs.size() == rhs.size());
    assert(lhs.size() == out.size());

    if constexpr (std::same_as<StorageType, std::int8_t>) {
#if defined(__SSE2__)
        auto const kernel = [](auto left, auto right) { return _mm_adds_epi8(left, right); };
        detail::apply_fixed_point_kernel_sse<8>(lhs, rhs, out, std::plus{}, kernel);
#elif defined(__ARM_NEON__)
        auto const kernel = [](int8x16_t left, int8x16_t right) { return vqaddq_s8(left, right); };
        detail::apply_fixed_point_kernel_neon128<8>(lhs, rhs, out, std::multiplies{}, kernel);
#else
        for (auto i{0U}; i < lhs.size(); ++i) { out[i] = std::plus{}(lhs[i], rhs[i]); }
#endif
    } else if constexpr (std::same_as<StorageType, std::int16_t>) {
#if defined(__SSE2__)
        auto const kernel = [](auto left, auto right) { return _mm_adds_epi16(left, right); };
        detail::apply_fixed_point_kernel_sse<16>(lhs, rhs, out, std::plus{}, kernel);
#elif defined(__ARM_NEON__)
        auto const kernel = [](int16x8_t left, int16x8_t right) { return vqaddq_s16(left, right); };
        detail::apply_fixed_point_kernel_neon128<16>(lhs, rhs, out, std::multiplies{}, kernel);
#else
        for (auto i{0U}; i < lhs.size(); ++i) { out[i] = std::plus{}(lhs[i], rhs[i]); }
#endif
    } else {
        for (auto i{0U}; i < lhs.size(); ++i) { out[i] = std::plus{}(lhs[i], rhs[i]); }
    }
}

/// out[i] = saturate16(lhs[i] - rhs[i])
template<int IntegerBits, int FractionalBits, typename StorageType, std::size_t Extent>
auto subtract(
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType> const, Extent> lhs,
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType> const, Extent> rhs,
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType>, Extent> out
)
{
    assert(lhs.size() == rhs.size());
    assert(lhs.size() == out.size());

    if constexpr (std::same_as<StorageType, std::int8_t>) {
#if defined(__SSE2__)
        auto const kernel = [](auto left, auto right) { return _mm_subs_epi8(left, right); };
        detail::apply_fixed_point_kernel_sse<8>(lhs, rhs, out, std::minus{}, kernel);
#elif defined(__ARM_NEON__)
        auto const kernel = [](int8x16_t left, int8x16_t right) { return vqsubq_s8(left, right); };
        detail::apply_fixed_point_kernel_neon128<8>(lhs, rhs, out, std::multiplies{}, kernel);
#else
        for (auto i{0U}; i < lhs.size(); ++i) { out[i] = std::minus{}(lhs[i], rhs[i]); }
#endif
    } else if constexpr (std::same_as<StorageType, std::int16_t>) {
#if defined(__SSE2__)
        auto const kernel = [](auto left, auto right) { return _mm_subs_epi16(left, right); };
        detail::apply_fixed_point_kernel_sse<16>(lhs, rhs, out, std::minus{}, kernel);
#elif defined(__ARM_NEON__)
        auto const kernel = [](int16x8_t left, int16x8_t right) { return vqsubq_s16(left, right); };
        detail::apply_fixed_point_kernel_neon128<16>(lhs, rhs, out, std::multiplies{}, kernel);
#else
        for (auto i{0U}; i < lhs.size(); ++i) { out[i] = std::minus{}(lhs[i], rhs[i]); }
#endif
    } else {
        for (auto i{0U}; i < lhs.size(); ++i) { out[i] = std::minus{}(lhs[i], rhs[i]); }
    }
}

/// out[i] = (lhs[i] * rhs[i]) >> FractionalBits;
template<int IntegerBits, int FractionalBits, typename StorageType, std::size_t Extent>
auto multiply(
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType> const, Extent> lhs,
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType> const, Extent> rhs,
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType>, Extent> out
)
{
    assert(lhs.size() == rhs.size());
    assert(lhs.size() == out.size());

    if constexpr (std::same_as<StorageType, std::int16_t> && FractionalBits == 15) {
#if defined(__SSE3__)
        auto const kernel = [](__m128i left, __m128i right) -> __m128i { return _mm_mulhrs_epi16(left, right); };
        detail::apply_fixed_point_kernel_sse<16>(lhs, rhs, out, std::multiplies{}, kernel);
#elif defined(__ARM_NEON__)
        auto const kernel = [](int16x8_t left, int16x8_t right) { return vqdmulhq_s16(left, right); };
        detail::apply_fixed_point_kernel_neon128<16>(lhs, rhs, out, std::multiplies{}, kernel);
#else
        for (auto i{0U}; i < lhs.size(); ++i) { out[i] = std::multiplies{}(lhs[i], rhs[i]); }
#endif
    } else {
        if constexpr (std::same_as<StorageType, std::int8_t>) {
#if defined(__SSE4_1__)
            auto const kernel = [](__m128i left, __m128i right) -> __m128i {
                auto const lowLeft    = _mm_cvtepi8_epi16(left);
                auto const lowRight   = _mm_cvtepi8_epi16(right);
                auto const lowProduct = _mm_mullo_epi16(lowLeft, lowRight);
                auto const lowShifted = _mm_srli_epi16(lowProduct, FractionalBits);

                auto const highLeft    = _mm_cvtepi8_epi16(_mm_srli_si128(left, 8));
                auto const highRight   = _mm_cvtepi8_epi16(_mm_srli_si128(right, 8));
                auto const highProduct = _mm_mullo_epi16(highLeft, highRight);
                auto const highShifted = _mm_srli_epi16(highProduct, FractionalBits);

                return _mm_packs_epi16(lowShifted, highShifted);
            };
            detail::apply_fixed_point_kernel_sse<8>(lhs, rhs, out, std::multiplies{}, kernel);
#else
            for (auto i{0U}; i < lhs.size(); ++i) { out[i] = std::multiplies{}(lhs[i], rhs[i]); }
#endif
        } else if constexpr (std::same_as<StorageType, std::int16_t>) {
#if defined(__SSE4_1__)
            auto const kernel = [](__m128i left, __m128i right) -> __m128i {
                auto const lowLeft    = _mm_cvtepi16_epi32(left);
                auto const lowRight   = _mm_cvtepi16_epi32(right);
                auto const lowProduct = _mm_mullo_epi32(lowLeft, lowRight);
                auto const lowShifted = _mm_srli_epi32(lowProduct, FractionalBits);

                auto const highLeft    = _mm_cvtepi16_epi32(_mm_srli_si128(left, 8));
                auto const highRight   = _mm_cvtepi16_epi32(_mm_srli_si128(right, 8));
                auto const highProduct = _mm_mullo_epi32(highLeft, highRight);
                auto const highShifted = _mm_srli_epi32(highProduct, FractionalBits);

                return _mm_packs_epi32(lowShifted, highShifted);
            };

            detail::apply_fixed_point_kernel_sse<16>(lhs, rhs, out, std::multiplies{}, kernel);
#elif defined(__ARM_NEON__)
            auto const kernel = [](int16x8_t left, int16x8_t right) { return vqdmulhq_s16(left, right); };
            detail::apply_fixed_point_kernel_neon128<8>(lhs, rhs, out, std::multiplies{}, kernel);
#else
            for (auto i{0U}; i < lhs.size(); ++i) { out[i] = std::multiplies{}(lhs[i], rhs[i]); }
#endif
        } else {
            for (auto i{0U}; i < lhs.size(); ++i) { out[i] = std::multiplies{}(lhs[i], rhs[i]); }
        }
    }
}

}  // namespace neo::fft
