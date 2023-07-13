#pragma once

#include "neo/convolution/container/mdspan.hpp"

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_graphics/juce_graphics.h>

#include <complex>

namespace neo::fft
{

[[nodiscard]] auto powerSpectrumImage(Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> frames,
                                      float threshold) -> juce::Image;

[[nodiscard]] auto powerSpectrumImage(juce::AudioBuffer<float> const& buffer, float threshold) -> juce::Image;

[[nodiscard]] auto powerHistogram(Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> frames)
    -> std::vector<int>;

[[nodiscard]] auto
powerHistogramImage(Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> spectogram) -> juce::Image;

}  // namespace neo::fft
