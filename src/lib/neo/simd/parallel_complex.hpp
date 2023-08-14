#pragma once

#include <neo/config.hpp>

#include <neo/math/complex.hpp>

namespace neo::simd {

template<typename FloatBatch>
struct alignas(FloatBatch::alignment) parallel_complex
{
    using batch_type       = FloatBatch;
    using register_type    = typename FloatBatch::register_type;
    using real_scalar_type = typename FloatBatch::value_type;

    static constexpr auto const size = FloatBatch::size;

    parallel_complex() noexcept = default;

    parallel_complex(FloatBatch real, FloatBatch imag) noexcept : _real{real}, _imag{imag} {}

    parallel_complex(register_type real, register_type imag) noexcept : _real{real}, _imag{imag} {}

    [[nodiscard]] NEO_ALWAYS_INLINE auto real() const noexcept -> FloatBatch { return _real; }

    [[nodiscard]] NEO_ALWAYS_INLINE auto imag() const noexcept -> FloatBatch { return _imag; }

    NEO_ALWAYS_INLINE friend auto operator+(parallel_complex lhs, parallel_complex rhs) -> parallel_complex
    {
        return parallel_complex{
            lhs.real() + rhs.real(),
            lhs.imag() + rhs.imag(),
        };
    }

    NEO_ALWAYS_INLINE friend auto operator-(parallel_complex lhs, parallel_complex rhs) -> parallel_complex
    {
        return parallel_complex{
            lhs.real() - rhs.real(),
            lhs.imag() - rhs.imag(),
        };
    }

    NEO_ALWAYS_INLINE friend auto operator*(parallel_complex lhs, parallel_complex rhs) -> parallel_complex
    {
        return parallel_complex{
            lhs.real() * rhs.real() - lhs.imag() * rhs.imag(),
            lhs.real() * rhs.imag() + lhs.imag() * rhs.real(),
        };
    }

private:
    FloatBatch _real;
    FloatBatch _imag;
};

}  // namespace neo::simd

template<typename FloatBatch>
inline constexpr auto const neo::is_complex<neo::simd::parallel_complex<FloatBatch>> = true;
