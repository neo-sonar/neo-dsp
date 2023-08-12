#pragma once

#include <neo/container/mdspan.hpp>
#include <neo/fft/algorithm/peak_normalize.hpp>

#include <juce_audio_basics/juce_audio_basics.h>

namespace neo {

auto juce_normalization(juce::AudioBuffer<float>& buf) -> void;

}  // namespace neo
