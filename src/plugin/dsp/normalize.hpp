#pragma once

#include <neo/algorithm/peak_normalize.hpp>
#include <neo/container/mdspan.hpp>

#include <juce_audio_basics/juce_audio_basics.h>

namespace neo {

auto juce_normalization(juce::AudioBuffer<float>& buf) -> void;

}  // namespace neo
