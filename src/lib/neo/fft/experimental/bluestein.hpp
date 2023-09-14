#pragma once

#include <neo/complex/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/fft.hpp>
#include <neo/math/bit_ceil.hpp>
#include <neo/math/ilog2.hpp>

#include <cstddef>
#include <numbers>

namespace neo::fft::experimental {

template<typename Complex>
struct bluestein_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    explicit bluestein_plan(size_type size) : _size{size} {}

    [[nodiscard]] auto size() const noexcept -> size_type { return _size; }

    template<inout_vector Vec>
        requires std::same_as<typename Vec::value_type, Complex>
    auto operator()(Vec x, direction dir) -> void
    {
        assert(std::cmp_equal(x.extent(0), size()));
        (void)(x);
        (void)(dir);
    }

private:
    size_type _size;
    fft_plan<Complex> _fft{bit_ceil(_size)};
};

auto dft(std::vector<double>& real, std::vector<double>& imag) -> void;
auto idft(std::vector<double>& real, std::vector<double>& imag) -> void;
auto radix2(std::vector<double>& real, std::vector<double>& imag) -> void;
auto bluestein(std::vector<double>& real, std::vector<double>& imag) -> void;

[[nodiscard]] auto
convolve(std::vector<double> xre, std::vector<double> xim, std::vector<double> yre, std::vector<double> yim)
    -> std::pair<std::vector<double>, std::vector<double> >;

inline auto radix2(std::vector<double>& real, std::vector<double>& imag) -> void
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
    auto const twoPi = std::numbers::pi * 2.0;

    // Trigonometric tables
    auto cosTable = std::vector<double>(n / 2);
    auto sinTable = std::vector<double>(n / 2);
    for (std::size_t i = 0; i < n / 2; ++i) {
        auto const x = twoPi * static_cast<double>(i) / static_cast<double>(n);
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
                double tpre   = real[l] * cosTable[k] + imag[l] * sinTable[k];
                double tpim   = -real[l] * sinTable[k] + imag[l] * cosTable[k];
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

inline auto dft(std::vector<double>& real, std::vector<double>& imag) -> void
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

inline auto idft(std::vector<double>& real, std::vector<double>& imag) -> void { dft(imag, real); }

inline auto bluestein(std::vector<double>& real, std::vector<double>& imag) -> void
{
    // Find a power-of-2 convolution length m such that m >= n * 2 + 1
    auto const n = real.size();
    auto const m = bit_ceil(n * 2U + 1U);

    // Trigonometric tables
    auto cosTable = std::vector<double>(n);
    auto sinTable = std::vector<double>(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto temp = static_cast<std::uintmax_t>(i) * i;
        temp %= static_cast<std::uintmax_t>(n) * 2;
        double angle = M_PI * static_cast<double>(temp) / static_cast<double>(n);
        cosTable[i]  = std::cos(angle);
        sinTable[i]  = std::sin(angle);
    }

    // Temporary vectors and preprocessing
    auto areal = std::vector<double>(m);
    auto aimag = std::vector<double>(m);
    for (std::size_t i = 0; i < n; ++i) {
        areal[i] = real[i] * cosTable[i] + imag[i] * sinTable[i];
        aimag[i] = -real[i] * sinTable[i] + imag[i] * cosTable[i];
    }

    auto breal = std::vector<double>(m);
    auto bimag = std::vector<double>(m);
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

inline auto convolve(std::vector<double> xre, std::vector<double> xim, std::vector<double> yre, std::vector<double> yim)
    -> std::pair<std::vector<double>, std::vector<double> >
{
    auto const n = xre.size();

    radix2(xre, xim);
    radix2(yre, yim);

    for (std::size_t i = 0; i < n; ++i) {
        double temp = xre[i] * yre[i] - xim[i] * yim[i];
        xim[i]      = xim[i] * yre[i] + xre[i] * yim[i];
        xre[i]      = temp;
    }

    radix2(xim, xre);

    for (std::size_t i = 0; i < n; ++i) {
        xre[i] /= static_cast<double>(n);
        xim[i] /= static_cast<double>(n);
    }

    return std::make_pair(std::move(xre), std::move(xim));
}

}  // namespace neo::fft::experimental
