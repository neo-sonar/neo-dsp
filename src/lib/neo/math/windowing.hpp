#pragma once

#include <neo/fft/container/mdspan.hpp>

#include <cmath>
#include <concepts>
#include <numbers>

namespace neo::fft {

template<std::floating_point Float>
[[nodiscard]] auto generate_hann_window(std::size_t length)
    -> KokkosEx::mdarray<Float, Kokkos::dextents<std::size_t, 1>>
{
    static constexpr auto const twoPi = static_cast<Float>(std::numbers::pi) * Float(2);

    auto window = KokkosEx::mdarray<Float, Kokkos::dextents<std::size_t, 1>>{length};
    for (auto i{0ULL}; i < window.extent(0); ++i) {
        auto const n = static_cast<Float>(length);
        window(i)    = Float(0.5) * (Float(1) - std::cos(twoPi * static_cast<Float>(i) / (n - Float(1))));
    }
    return window;
}
}  // namespace neo::fft
