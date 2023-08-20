#pragma once

#include <neo/complex/complex.hpp>

namespace neo {

template<typename Scalar>
struct scalar_complex
{
    using value_type = Scalar;

    constexpr scalar_complex() = default;

    constexpr scalar_complex(Scalar re, Scalar im = Scalar{}) noexcept : _data{re, im} {}

    [[nodiscard]] constexpr auto real() const noexcept -> Scalar { return _data[0]; }

    [[nodiscard]] constexpr auto imag() const noexcept -> Scalar { return _data[1]; }

    constexpr auto real(Scalar re) noexcept -> void { _data[0] = re; }

    constexpr auto imag(Scalar im) noexcept -> void { _data[1] = im; }

    friend constexpr auto operator+(scalar_complex lhs, scalar_complex rhs) noexcept -> scalar_complex
    {
        return scalar_complex{
            lhs.real() + rhs.real(),
            lhs.imag() + rhs.imag(),
        };
    }

    friend constexpr auto operator-(scalar_complex lhs, scalar_complex rhs) noexcept -> scalar_complex
    {
        return scalar_complex{
            lhs.real() - rhs.real(),
            lhs.imag() - rhs.imag(),
        };
    }

    friend constexpr auto operator*(scalar_complex lhs, scalar_complex rhs) noexcept -> scalar_complex
    {
        return scalar_complex{
            lhs.real() * rhs.real() - lhs.imag() * rhs.imag(),
            lhs.real() * rhs.imag() + lhs.imag() * rhs.real(),
        };
    }

    template<typename OtherScalar>
        requires(not complex<OtherScalar>)
    friend constexpr auto operator*=(scalar_complex& lhs, OtherScalar rhs) noexcept -> scalar_complex&
    {
        lhs = scalar_complex{lhs.real() * rhs, lhs.imag()};
        return lhs;
    }

private:
    Scalar _data[2];  // NOLINT
};

template<typename Scalar>
inline constexpr auto const is_complex<scalar_complex<Scalar>> = true;

template<typename Scalar>
[[nodiscard]] constexpr auto conj(scalar_complex<Scalar> const& z) noexcept -> scalar_complex<Scalar>
{
    return {z.real(), -z.imag()};
}

template<typename Scalar>
[[nodiscard]] constexpr auto abs(scalar_complex<Scalar> const& z) noexcept -> Scalar
{
    auto const re = z.real();
    auto const im = z.imag();
    return std::sqrt(re * re + im * im);
}

using complex64  = scalar_complex<float>;
using complex128 = scalar_complex<double>;

}  // namespace neo
