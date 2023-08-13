#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>

#include <arm_neon.h>

namespace neo::simd {

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

    static constexpr auto vectorSize = static_cast<std::ptrdiff_t>(128 / ValueSizeBits);
    auto const remainder             = static_cast<std::ptrdiff_t>(lhs.size()) % vectorSize;

    for (auto i{0}; i < remainder; ++i) {
        out[static_cast<size_t>(i)] = scalar_kernel(lhs[static_cast<size_t>(i)], rhs[static_cast<size_t>(i)]);
    }

    for (auto i{remainder}; i < std::ssize(lhs); i += vectorSize) {
        auto const left  = load(std::next(lhs.data(), i));
        auto const right = load(std::next(rhs.data(), i));
        store(std::next(out.data(), i), vector_kernel(left, right));
    }
};

}  // namespace neo::simd
