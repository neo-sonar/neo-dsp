#pragma once

#include <complex>
#include <cstddef>
#include <numbers>
#include <span>

namespace neo::fft {

template<typename T>
auto dft(std::span<std::complex<T> const> in, std::span<std::complex<T>> out) -> void
{
    static constexpr auto const pi = static_cast<T>(std::numbers::pi);

    auto const N = in.size();
    for (std::size_t k = 0; k < N; ++k) {
        auto tmp = std::complex<T>{};
        for (std::size_t n = 0; n < N; ++n) { tmp += in[n] * std::polar(T(1), T(-2) * pi * n * k / N); }
        out[k] = tmp;
    }
}

}  // namespace neo::fft
