#pragma once

#include "neo/mdspan.hpp"

#include <juce_dsp/juce_dsp.h>
#include <juce_graphics/juce_graphics.h>

#include <complex>
#include <span>

namespace neo
{

auto rfft(juce::dsp::FFT& fft, std::span<float const> input, std::span<std::complex<float>> output) -> void;
auto irfft(juce::dsp::FFT& fft, std::span<std::complex<float> const> input, std::span<float> output) -> void;

[[nodiscard]] auto stft(juce::AudioBuffer<float> const& buffer, int windowSize)
    -> KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>;

[[nodiscard]] auto powerSpectrumImage(Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> frames,
                                      float threshold) -> juce::Image;

[[nodiscard]] auto powerSpectrumImage(juce::AudioBuffer<float> const& buffer, float threshold) -> juce::Image;

[[nodiscard]] auto powerHistogram(Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> frames)
    -> std::vector<int>;

[[nodiscard]] auto
powerHistogramImage(Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> spectogram) -> juce::Image;

}  // namespace neo
