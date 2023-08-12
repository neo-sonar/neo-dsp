#pragma once

#include "neo/fft.hpp"
#include "neo/fft/convolution.hpp"

#include <algorithm>
#include <juce_dsp/juce_dsp.h>
#include <vector>

namespace neo {

[[nodiscard]] auto dense_convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter)
    -> juce::AudioBuffer<float>;

[[nodiscard]] auto
sparse_convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter, float thresholdDB)
    -> juce::AudioBuffer<float>;

}  // namespace neo
