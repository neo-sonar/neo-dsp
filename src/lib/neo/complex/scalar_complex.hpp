#pragma once

#include <neo/complex/complex.hpp>

#include <array>

namespace neo {

template<typename ScalarType>
struct scalar_complex
{
    using value_type = ScalarType;

    constexpr scalar_complex() = default;

    constexpr scalar_complex(ScalarType re, ScalarType im = ScalarType{}) noexcept : _data{re, im} {}

    [[nodiscard]] constexpr auto real() const noexcept -> ScalarType { return _data[0]; }

    [[nodiscard]] constexpr auto imag() const noexcept -> ScalarType { return _data[1]; }

    constexpr auto real(ScalarType re) noexcept -> void { _data[0] = re; }

    constexpr auto imag(ScalarType im) noexcept -> void { _data[1] = im; }

    friend constexpr auto operator+(scalar_complex lhs, scalar_complex rhs) -> scalar_complex
    {
        return scalar_complex{
            lhs.real() + rhs.real(),
            lhs.imag() + rhs.imag(),
        };
    }

    friend constexpr auto operator-(scalar_complex lhs, scalar_complex rhs) -> scalar_complex
    {
        return scalar_complex{
            lhs.real() - rhs.real(),
            lhs.imag() - rhs.imag(),
        };
    }

    friend constexpr auto operator*(scalar_complex lhs, scalar_complex rhs) -> scalar_complex
    {
        return scalar_complex{
            lhs.real() * rhs.real() - lhs.imag() * rhs.imag(),
            lhs.real() * rhs.imag() + lhs.imag() * rhs.real(),
        };
    }

private:
    std::array<ScalarType, 2> _data;  // NOLINT
};

template<typename ScalarType>
inline constexpr auto const is_complex<scalar_complex<ScalarType>> = true;

using complex64  = scalar_complex<float>;
using complex128 = scalar_complex<double>;

}  // namespace neo
