#pragma once

#include <concepts>
#include <juce_audio_basics/juce_audio_basics.h>
#include <span>

namespace neo {

// normalized_sample = sample / max(abs(buffer))
auto peak_normalization(std::span<float> buffer) -> void;

// normalized_sample = sample / sqrt(mean(buffer^2))
auto rms_normalization(std::span<float> buffer) -> void;

auto juce_normalization(juce::AudioBuffer<float>& buf) -> void;

}  // namespace neo
