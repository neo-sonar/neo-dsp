#pragma once

#include <juce_audio_formats/juce_audio_formats.h>

namespace neo
{

auto resample(juce::AudioBuffer<float> const& buf, double srcSampleRate, double destSampleRate)
    -> juce::AudioBuffer<float>;

[[nodiscard]] auto loadAndResample(juce::AudioFormatManager& formats, juce::File const& file, double sampleRate)
    -> juce::AudioBuffer<float>;

}  // namespace neo
