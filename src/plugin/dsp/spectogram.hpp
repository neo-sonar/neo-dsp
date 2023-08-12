#pragma once

#include <neo/fft/container/mdspan.hpp>
#include <neo/math/complex.hpp>

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_graphics/juce_graphics.h>

namespace neo::fft {

[[nodiscard]] auto powerSpectrumImage(
    Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> frames,
    std::function<float(std::size_t)> const& weighting,
    float threshold
) -> juce::Image;

[[nodiscard]] auto powerHistogram(
    Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> spectogram,
    std::function<float(std::size_t)> const& weighting
) -> std::vector<int>;

[[nodiscard]] auto powerHistogramImage(
    Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> spectogram,
    std::function<float(std::size_t)> const& weighting
) -> juce::Image;

}  // namespace neo::fft
