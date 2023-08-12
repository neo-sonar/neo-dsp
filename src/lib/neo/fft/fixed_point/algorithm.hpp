#pragma once

#include <neo/config.hpp>

#include <neo/fft/fixed_point/fixed_point.hpp>

#if defined(__SSE2__)
    #include <emmintrin.h>
    #include <smmintrin.h>
#endif

#if defined(__SSE3__)
    #include <immintrin.h>
    #include <tmmintrin.h>
#endif

#if defined(__ARM_NEON__)
    #include <arm_neon.h>
#endif

#include <functional>
#include <iterator>
#include <span>

namespace neo::fft {

namespace detail {
#if defined(__SSE2__)
template<int ValueSizeBits>
inline constexpr auto apply_kernel_sse
    = [](auto const& lhs, auto const& rhs, auto const& out, auto scalar_kernel, auto vector_kernel) {
    static constexpr auto vectorSize = static_cast<ptrdiff_t>(128 / ValueSizeBits);
    auto const remainder             = static_cast<ptrdiff_t>(lhs.size()) % vectorSize;

    for (auto i{0}; i < remainder; ++i) {
        out[static_cast<size_t>(i)] = scalar_kernel(lhs[static_cast<size_t>(i)], rhs[static_cast<size_t>(i)]);
    }

    for (auto i{remainder}; i < std::ssize(lhs); i += vectorSize) {
        auto const left  = _mm_loadu_si128(reinterpret_cast<__m128i const*>(std::next(lhs.data(), i)));
        auto const right = _mm_loadu_si128(reinterpret_cast<__m128i const*>(std::next(rhs.data(), i)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(std::next(out.data(), i)), vector_kernel(left, right));
    }
};
#endif

#if defined(__ARM_NEON__)
template<int ValueSizeBits>
inline constexpr auto apply_kernel_neon128
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

    for (auto i{remainder}; i < std::ssize(lhs); i += vectorSize) {
        auto const left  = load(std::next(lhs.data(), i));
        auto const right = load(std::next(rhs.data(), i));
        store(std::next(out.data(), i), vector_kernel(left, right));
    }
};
#endif
}  // namespace detail

/// out[i] = saturate16(lhs[i] + rhs[i])
template<int IntegerBits, int FractionalBits, typename StorageType, std::size_t Extent>
auto add(
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType> const, Extent> lhs,
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType> const, Extent> rhs,
    std::span<fixed_point<IntegerBits, FractionalBits, StorageType>, Extent> out
)
{
    NEO_FFT_PRECONDITION(lhs.size() == rhs.size());
    NEO_FFT_PRECONDITION(lhs.size() == out.size());

    if constexpr (std::same_as<StorageType, std::int8_t>) {
#if defined(__SSE2__)
        auto const kernel = [](auto left, auto right) { return _mm_adds_epi8(left, right); };
        detail::apply_kernel_sse<8>(lhs, rhs, out, std::plus{}, kernel);
#elif defined(__ARM_NEON__)
        auto const kernel = [](int8x16_t left, int8x16_t right) { return vqaddq_s8(left, right); };
        detail::apply_kernel_neon128<8>(lhs, rhs, out, std::plus{}, kernel);
#else
        for (auto i{0U}; i < lhs.size(); ++i) {
            out[i] = std::plus{}(lhs[i], rhs[i]);
        }
#endif
    } else if constexpr (std::same_as<StorageType, std::int16_t>) {
#if defined(__SSE2__)
        auto const kernel = [](auto left, auto right) { return _mm_adds_epi16(left, right); };
        detail::apply_kernel_sse<16>(lhs, rhs, out, std::plus{}, kernel);
#elif defined(__ARM_NEON__)
        auto const kernel = [](int16x8_t left, int16x8_t right) { return vqaddq_s16(left, right); };
        detail::apply_kernel_neon128<16>(lhs, rhs, out, std::plus{}, kernel);
#else
        for (auto i{0U}; i < lhs.size(); ++i) {
            out[i] = std::plus{}(lhs[i], rhs[i]);
        }
#endif
    } else {
        for (auto i{0U}; i < lhs.size(); ++i) {
            out[i] = std::plus{}(lhs[i], rhs[i]);
        }
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
    NEO_FFT_PRECONDITION(lhs.size() == rhs.size());
    NEO_FFT_PRECONDITION(lhs.size() == out.size());

    if constexpr (std::same_as<StorageType, std::int8_t>) {
#if defined(__SSE2__)
        auto const kernel = [](auto left, auto right) { return _mm_subs_epi8(left, right); };
        detail::apply_kernel_sse<8>(lhs, rhs, out, std::minus{}, kernel);
#elif defined(__ARM_NEON__)
        auto const kernel = [](int8x16_t left, int8x16_t right) { return vqsubq_s8(left, right); };
        detail::apply_kernel_neon128<8>(lhs, rhs, out, std::minus{}, kernel);
#else
        for (auto i{0U}; i < lhs.size(); ++i) {
            out[i] = std::minus{}(lhs[i], rhs[i]);
        }
#endif
    } else if constexpr (std::same_as<StorageType, std::int16_t>) {
#if defined(__SSE2__)
        auto const kernel = [](auto left, auto right) { return _mm_subs_epi16(left, right); };
        detail::apply_kernel_sse<16>(lhs, rhs, out, std::minus{}, kernel);
#elif defined(__ARM_NEON__)
        auto const kernel = [](int16x8_t left, int16x8_t right) { return vqsubq_s16(left, right); };
        detail::apply_kernel_neon128<16>(lhs, rhs, out, std::minus{}, kernel);
#else
        for (auto i{0U}; i < lhs.size(); ++i) {
            out[i] = std::minus{}(lhs[i], rhs[i]);
        }
#endif
    } else {
        for (auto i{0U}; i < lhs.size(); ++i) {
            out[i] = std::minus{}(lhs[i], rhs[i]);
        }
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
    NEO_FFT_PRECONDITION(lhs.size() == rhs.size());
    NEO_FFT_PRECONDITION(lhs.size() == out.size());

    if constexpr (std::same_as<StorageType, std::int16_t> && FractionalBits == 15) {
#if defined(__SSE3__)
        auto const kernel = [](__m128i left, __m128i right) -> __m128i { return _mm_mulhrs_epi16(left, right); };
        detail::apply_kernel_sse<16>(lhs, rhs, out, std::multiplies{}, kernel);
#elif defined(__ARM_NEON__)
        auto const kernel = [](int16x8_t left, int16x8_t right) { return vqdmulhq_s16(left, right); };
        detail::apply_kernel_neon128<16>(lhs, rhs, out, std::multiplies{}, kernel);
#else
        for (auto i{0U}; i < lhs.size(); ++i) {
            out[i] = std::multiplies{}(lhs[i], rhs[i]);
        }
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
            detail::apply_kernel_sse<8>(lhs, rhs, out, std::multiplies{}, kernel);
#else
            for (auto i{0U}; i < lhs.size(); ++i) {
                out[i] = std::multiplies{}(lhs[i], rhs[i]);
            }
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

            detail::apply_kernel_sse<16>(lhs, rhs, out, std::multiplies{}, kernel);
#elif defined(__ARM_NEON__)
            auto const kernel = [](int16x8_t left, int16x8_t right) { return vqdmulhq_s16(left, right); };
            detail::apply_kernel_neon128<8>(lhs, rhs, out, std::multiplies{}, kernel);
#else
            for (auto i{0U}; i < lhs.size(); ++i) {
                out[i] = std::multiplies{}(lhs[i], rhs[i]);
            }
#endif
        } else {
            for (auto i{0U}; i < lhs.size(); ++i) {
                out[i] = std::multiplies{}(lhs[i], rhs[i]);
            }
        }
    }
}

}  // namespace neo::fft
