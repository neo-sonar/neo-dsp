#pragma once

#include <neo/math/complex.hpp>
#include <neo/math/fixed_point/fixed_point.hpp>

#include <array>
#include <cstdint>

namespace neo {

struct complex_q15
{
    using value_type = q15;

    constexpr complex_q15() = default;

    constexpr complex_q15(q15 re, q15 im = q15{}) noexcept : _data{re, im} {}

    [[nodiscard]] constexpr auto real() const noexcept -> q15 { return _data[0]; }

    [[nodiscard]] constexpr auto imag() const noexcept -> q15 { return _data[1]; }

    constexpr auto real(q15 re) noexcept -> void { _data[0] = re; }

    constexpr auto imag(q15 im) noexcept -> void { _data[1] = im; }

    friend constexpr auto operator+(complex_q15 lhs, complex_q15 rhs) -> complex_q15
    {
        return complex_q15{lhs.real() + rhs.real(), lhs.imag() + rhs.imag()};
    }

    friend constexpr auto operator-(complex_q15 lhs, complex_q15 rhs) -> complex_q15
    {
        return complex_q15{lhs.real() - rhs.real(), lhs.imag() - rhs.imag()};
    }

    friend constexpr auto operator*(complex_q15 lhs, complex_q15 rhs) -> complex_q15
    {
        auto const a = lhs.real().value();
        auto const b = lhs.imag().value();
        auto const c = rhs.real().value();
        auto const d = lhs.imag().value();

        return complex_q15{
            {underlying_value, static_cast<std::int16_t>(((a * c) >> 17) - ((b * d) >> 17))},
            {underlying_value, static_cast<std::int16_t>(((a * d) >> 17) + ((b * c) >> 17))},
        };
    }

private:
    std::array<q15, 2> _data{};
};

template<>
inline constexpr auto const is_complex<complex_q15> = true;

}  // namespace neo
