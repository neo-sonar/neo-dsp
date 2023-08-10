#pragma once

#include <neo/fft/algorithm/peak_normalize.hpp>
#include <neo/fft/container/mdspan.hpp>

#include <juce_audio_basics/juce_audio_basics.h>

namespace neo {

// normalized_sample = sample / sqrt(mean(buffer^2))
auto rms_normalization(std::span<float> buffer) -> void;

auto juce_normalization(juce::AudioBuffer<float>& buf) -> void;

}  // namespace neo
