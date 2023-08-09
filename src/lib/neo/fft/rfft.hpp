#pragma once

#include <complex>
#include <span>

namespace neo::fft {

template<typename Float>
auto extract_two_real_dfts(
    std::span<std::complex<Float> const> dft,
    std::span<std::complex<Float>> a,
    std::span<std::complex<Float>> b
) -> void
{
    auto const n = dft.size();
    auto const i = std::complex{Float(0), Float(-1)};

    a[0] = dft[0].real();
    b[0] = dft[0].imag();

    for (auto k{1U}; k < n / 2 + 1; ++k) {
        a[k] = (dft[k] + std::conj(dft[n - k])) * Float(0.5);
        b[k] = (i * (dft[k] - std::conj(dft[n - k]))) * Float(0.5);
    }
}

}  // namespace neo::fft
