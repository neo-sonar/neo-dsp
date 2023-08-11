#pragma once

#include <neo/fft/config.hpp>

#include <neo/fft/container/mdspan.hpp>

#include <complex>
#include <numbers>

namespace neo::fft {

template<in_vector InVec, out_vector OutVec>
    requires(std::same_as<typename InVec::value_type, typename OutVec::value_type>)
auto dft(InVec in, OutVec out) -> void
{
    NEO_FFT_PRECONDITION(in.extents() == out.extents());

    using Complex = typename OutVec::value_type;
    using Float   = typename Complex::value_type;

    static constexpr auto const pi = static_cast<Float>(std::numbers::pi);

    auto const N = in.extent(0);
    for (std::size_t k = 0; k < N; ++k) {
        auto tmp = Complex{};
        for (std::size_t n = 0; n < N; ++n) {
            using std::polar;
            auto const input = in(n);
            auto const w     = std::polar(Float(1), Float(-2) * pi * Float(n) * Float(k) / Float(N));
            tmp += input * w;
        }
        out(k) = tmp;
    }
}

}  // namespace neo::fft
