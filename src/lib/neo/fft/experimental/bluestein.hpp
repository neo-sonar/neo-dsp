#pragma once

#include <neo/complex/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/fft.hpp>
#include <neo/math/bit_ceil.hpp>
#include <neo/math/ilog2.hpp>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <numbers>

namespace neo::fft::experimental {

namespace detail {

template<std::floating_point Float>
auto dft(std::vector<Float>& real, std::vector<Float>& imag) -> void;

template<std::floating_point Float>
auto idft(std::vector<Float>& real, std::vector<Float>& imag) -> void;

template<std::floating_point Float>
auto radix2(std::vector<Float>& real, std::vector<Float>& imag) -> void;

template<std::floating_point Float>
auto bluestein(std::vector<Float>& real, std::vector<Float>& imag) -> void;

template<std::floating_point Float>
[[nodiscard]] auto
convolve(std::vector<Float> xre, std::vector<Float> xim, std::vector<Float> yre, std::vector<Float> yim)
    -> std::pair<std::vector<Float>, std::vector<Float> >;

template<std::floating_point Float>
auto radix2(std::vector<Float>& real, std::vector<Float>& imag) -> void
{
    auto reverseBits = [](std::size_t val, int width) -> std::size_t {
        std::size_t result = 0;
        for (int i = 0; i < width; ++i, val >>= 1)
            result = (result << 1) | (val & 1U);
        return result;
    };

    // Length variables
    auto const n     = real.size();
    auto const order = static_cast<int>(ilog2(n));
    auto const twoPi = static_cast<Float>(std::numbers::pi * 2.0);

    // Trigonometric tables
    auto cosTable = std::vector<Float>(n / 2);
    auto sinTable = std::vector<Float>(n / 2);
    for (std::size_t i = 0; i < n / 2; ++i) {
        auto const x = twoPi * static_cast<Float>(i) / static_cast<Float>(n);
        cosTable[i]  = std::cos(x);
        sinTable[i]  = std::sin(x);
    }

    // Bit-reversed addressing permutation
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t j = reverseBits(i, order);
        if (j > i) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }

    // Cooley-Tukey decimation-in-time radix-2 FFT
    for (std::size_t size = 2; size <= n; size *= 2) {
        std::size_t halfsize  = size / 2;
        std::size_t tablestep = n / size;
        for (std::size_t i = 0; i < n; i += size) {
            for (std::size_t j = i, k = 0; j < i + halfsize; j++, k += tablestep) {
                std::size_t l = j + halfsize;
                Float tpre    = real[l] * cosTable[k] + imag[l] * sinTable[k];
                Float tpim    = -real[l] * sinTable[k] + imag[l] * cosTable[k];
                real[l]       = real[j] - tpre;
                imag[l]       = imag[j] - tpim;
                real[j] += tpre;
                imag[j] += tpim;
            }
        }

        // Prevent overflow in 'size *= 2'
        if (size == n) {
            break;
        }
    }
}

template<std::floating_point Float>
auto dft(std::vector<Float>& real, std::vector<Float>& imag) -> void
{
    auto const n = real.size();
    if (n == 0) {
        return;
    }

    if ((n & (n - 1)) == 0) {
        radix2(real, imag);
    } else {
        bluestein(real, imag);
    }
}

template<std::floating_point Float>
auto idft(std::vector<Float>& real, std::vector<Float>& imag) -> void
{
    dft(imag, real);
}

template<std::floating_point Float>
auto bluestein(std::vector<Float>& real, std::vector<Float>& imag) -> void
{
    // Find a power-of-2 convolution length m such that m >= n * 2 + 1
    auto const n = real.size();
    auto const m = bit_ceil(n * 2U + 1U);

    // Trigonometric tables
    auto cosTable = std::vector<Float>(n);
    auto sinTable = std::vector<Float>(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto temp = static_cast<std::uintmax_t>(i) * i;
        temp %= static_cast<std::uintmax_t>(n) * 2;

        auto const pi    = static_cast<Float>(std::numbers::pi);
        auto const angle = pi * static_cast<Float>(temp) / static_cast<Float>(n);

        cosTable[i] = std::cos(angle);
        sinTable[i] = std::sin(angle);
    }

    // Temporary vectors and preprocessing
    auto areal = std::vector<Float>(m);
    auto aimag = std::vector<Float>(m);
    for (std::size_t i = 0; i < n; ++i) {
        areal[i] = real[i] * cosTable[i] + imag[i] * sinTable[i];
        aimag[i] = -real[i] * sinTable[i] + imag[i] * cosTable[i];
    }

    auto breal = std::vector<Float>(m);
    auto bimag = std::vector<Float>(m);
    breal[0]   = cosTable[0];
    bimag[0]   = sinTable[0];
    for (std::size_t i = 1; i < n; ++i) {
        breal[i] = breal[m - i] = cosTable[i];
        bimag[i] = bimag[m - i] = sinTable[i];
    }

    // Convolution
    auto [creal, cimag] = convolve(areal, aimag, breal, bimag);

    // Postprocessing
    for (std::size_t i = 0; i < n; ++i) {
        real[i] = creal[i] * cosTable[i] + cimag[i] * sinTable[i];
        imag[i] = -creal[i] * sinTable[i] + cimag[i] * cosTable[i];
    }
}

template<std::floating_point Float>
auto convolve(std::vector<Float> xre, std::vector<Float> xim, std::vector<Float> yre, std::vector<Float> yim)
    -> std::pair<std::vector<Float>, std::vector<Float> >
{
    auto const n = xre.size();

    radix2(xre, xim);
    radix2(yre, yim);

    for (std::size_t i = 0; i < n; ++i) {
        Float temp = xre[i] * yre[i] - xim[i] * yim[i];
        xim[i]     = xim[i] * yre[i] + xre[i] * yim[i];
        xre[i]     = temp;
    }

    radix2(xim, xre);

    for (std::size_t i = 0; i < n; ++i) {
        xre[i] /= static_cast<Float>(n);
        xim[i] /= static_cast<Float>(n);
    }

    return std::make_pair(std::move(xre), std::move(xim));
}

}  // namespace detail

template<typename Complex>
struct bluestein_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    explicit bluestein_plan(size_type size) : _size{size}, _real(size), _imag(size) {}

    [[nodiscard]] auto size() const noexcept -> size_type { return _size; }

    template<inout_vector Vec>
        requires std::same_as<typename Vec::value_type, Complex>
    auto operator()(Vec x, direction dir) -> void
    {
        assert(std::cmp_equal(x.extent(0), size()));

        if (dir == direction::forward) {
            for (size_type i{0}; i < size(); ++i) {
                _real[i] = x[i].real();
                _imag[i] = x[i].imag();
            }
        } else {
            for (size_type i{0}; i < size(); ++i) {
                _real[i] = x[i].imag();
                _imag[i] = x[i].real();
            }
        }

        detail::bluestein(_real, _imag);

        if (dir == direction::forward) {
            for (size_type i{0}; i < size(); ++i) {
                x[i] = Complex{_real[i], _imag[i]};
            }
        } else {
            for (size_type i{0}; i < size(); ++i) {
                x[i] = Complex{_imag[i], _real[i]};
            }
        }
    }

private:
    using real_type = typename Complex::value_type;

    size_type _size;
    fft_plan<Complex> _fft{bit_ceil(_size)};
    std::vector<real_type> _real;
    std::vector<real_type> _imag;
};

}  // namespace neo::fft::experimental
