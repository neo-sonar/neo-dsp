#pragma once

#include <juce_audio_formats/juce_audio_formats.h>

namespace neo
{

auto resample(juce::AudioBuffer<float> const& buf, double const srcSampleRate, double const destSampleRate)
    -> juce::AudioBuffer<float>;

}  // namespace neo
