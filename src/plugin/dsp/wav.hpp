#pragma once

#include <neo/container/mdspan.hpp>

#include <juce_audio_formats/juce_audio_formats.h>

namespace neo {

auto writeToWavFile(
    juce::File const& file,
    stdex::mdspan<float, stdex::dextents<size_t, 2>> buffer,
    double sampleRate,
    int bitsPerSample
) -> void;

auto writeToWavFile(
    juce::File const& file,
    juce::AudioBuffer<float> const& buffer,
    double sampleRate,
    int bitsPerSample
) -> void;

}  // namespace neo
