#pragma once

#include <neo/config.hpp>

#include <neo/math/complex.hpp>

#include <concepts>

namespace neo::simd {

template<typename FloatBatch>
struct alignas(FloatBatch::alignment) interleaved_complex
{
    using batch_type       = FloatBatch;
    using register_type    = typename FloatBatch::register_type;
    using real_scalar_type = typename FloatBatch::value_type;

    static constexpr auto const size = FloatBatch::size / 2U;

    interleaved_complex() noexcept = default;

    interleaved_complex(FloatBatch batch) noexcept : _batch{batch} {}

    interleaved_complex(register_type reg) noexcept : _batch{reg} {}

    template<neo::complex Complex>
        requires std::same_as<typename Complex::value_type, real_scalar_type>
    [[nodiscard]] static auto load_unaligned(Complex const* val) -> interleaved_complex
    {
        return batch_type::load_unaligned(reinterpret_cast<real_scalar_type const*>(val));
    }

    template<neo::complex Complex>
        requires std::same_as<typename Complex::value_type, real_scalar_type>
    auto store_unaligned(Complex* output) const -> void
    {
        return _batch.store_unaligned(reinterpret_cast<real_scalar_type*>(output));
    }

    [[nodiscard]] NEO_ALWAYS_INLINE auto batch() const -> batch_type { return _batch; }

    NEO_ALWAYS_INLINE friend auto operator+(interleaved_complex lhs, interleaved_complex rhs) noexcept
        -> interleaved_complex
    {
        return interleaved_complex{
            cadd(static_cast<register_type>(lhs.batch()), static_cast<register_type>(rhs.batch())),
        };
    }

    NEO_ALWAYS_INLINE friend auto operator-(interleaved_complex lhs, interleaved_complex rhs) noexcept
        -> interleaved_complex
    {
        return interleaved_complex{
            csub(static_cast<register_type>(lhs.batch()), static_cast<register_type>(rhs.batch())),
        };
    }

    NEO_ALWAYS_INLINE friend auto operator*(interleaved_complex lhs, interleaved_complex rhs) noexcept
        -> interleaved_complex
    {
        return interleaved_complex{
            cmul(static_cast<register_type>(lhs.batch()), static_cast<register_type>(rhs.batch())),
        };
    }

private:
    FloatBatch _batch;
};

}  // namespace neo::simd

template<typename FloatBatch>
inline constexpr auto const neo::is_complex<neo::simd::interleaved_complex<FloatBatch>> = true;
