#pragma once

#include "neo/fft/convolution/math/fixed_point/fixed_point.hpp"

#include <array>
#include <cstdint>

namespace neo::fft {

struct complex_q15_t
{
    using value_type = q15_t;

    constexpr complex_q15_t() = default;

    constexpr complex_q15_t(q15_t re, q15_t im = q15_t{}) noexcept : _data{re, im} {}

    [[nodiscard]] constexpr auto real() const noexcept -> q15_t { return _data[0]; }

    [[nodiscard]] constexpr auto imag() const noexcept -> q15_t { return _data[1]; }

    constexpr auto real(q15_t re) noexcept -> void { _data[0] = re; }

    constexpr auto imag(q15_t im) noexcept -> void { _data[1] = im; }

    friend constexpr auto operator+(complex_q15_t lhs, complex_q15_t rhs) -> complex_q15_t
    {
        return complex_q15_t{lhs.real() + rhs.real(), lhs.imag() + rhs.imag()};
    }

    friend constexpr auto operator-(complex_q15_t lhs, complex_q15_t rhs) -> complex_q15_t
    {
        return complex_q15_t{lhs.real() - rhs.real(), lhs.imag() - rhs.imag()};
    }

    friend constexpr auto operator*(complex_q15_t lhs, complex_q15_t rhs) -> complex_q15_t
    {
        auto const a = lhs.real().value();
        auto const b = lhs.imag().value();
        auto const c = rhs.real().value();
        auto const d = lhs.imag().value();

        return complex_q15_t{
            {underlying_value, static_cast<std::int16_t>(((a * c) >> 17) - ((b * d) >> 17))},
            {underlying_value, static_cast<std::int16_t>(((a * d) >> 17) + ((b * c) >> 17))},
        };
    }

private:
    std::array<q15_t, 2> _data{};
};

}  // namespace neo::fft
