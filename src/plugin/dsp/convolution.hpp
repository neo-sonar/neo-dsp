#pragma once

#include "neo/convolution.hpp"
#include "neo/fft.hpp"

#include <juce_dsp/juce_dsp.h>

#include <algorithm>
#include <vector>

namespace neo::fft
{

[[nodiscard]] auto dense_convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter)
    -> juce::AudioBuffer<float>;

[[nodiscard]] auto dense_stereo_convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter)
    -> juce::AudioBuffer<float>;

[[nodiscard]] auto sparse_convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter,
                                   float thresholdDB) -> juce::AudioBuffer<float>;

}  // namespace neo::fft
