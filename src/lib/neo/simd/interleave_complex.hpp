#pragma once

#include <neo/config.hpp>

#include <neo/math/complex.hpp>

#include <concepts>

namespace neo {

template<typename FloatBatch>
struct alignas(FloatBatch::alignment) interleave_complex
{
    using batch_type       = FloatBatch;
    using register_type    = typename FloatBatch::register_type;
    using real_scalar_type = typename FloatBatch::value_type;

    static constexpr auto const size = FloatBatch::size / 2U;

    interleave_complex() noexcept = default;

    interleave_complex(FloatBatch batch) noexcept : _batch{batch} {}

    interleave_complex(register_type reg) noexcept : _batch{reg} {}

    template<neo::complex Complex>
        requires std::same_as<typename Complex::value_type, real_scalar_type>
    [[nodiscard]] static auto load_unaligned(Complex const* val) -> interleave_complex
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

    NEO_ALWAYS_INLINE friend auto operator+(interleave_complex lhs, interleave_complex rhs) noexcept
        -> interleave_complex
    {
        return interleave_complex{
            cadd(static_cast<register_type>(lhs.batch()), static_cast<register_type>(rhs.batch())),
        };
    }

    NEO_ALWAYS_INLINE friend auto operator-(interleave_complex lhs, interleave_complex rhs) noexcept
        -> interleave_complex
    {
        return interleave_complex{
            csub(static_cast<register_type>(lhs.batch()), static_cast<register_type>(rhs.batch())),
        };
    }

    NEO_ALWAYS_INLINE friend auto operator*(interleave_complex lhs, interleave_complex rhs) noexcept
        -> interleave_complex
    {
        return interleave_complex{
            cmul(static_cast<register_type>(lhs.batch()), static_cast<register_type>(rhs.batch())),
        };
    }

private:
    FloatBatch _batch;
};

}  // namespace neo

template<typename FloatBatch>
inline constexpr auto const neo::is_complex<neo::interleave_complex<FloatBatch>> = true;
