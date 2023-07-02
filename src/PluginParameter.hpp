#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_data_structures/juce_data_structures.h>

namespace mc
{

struct ParamID
{
    // i/o
    static constexpr char const* inGain  = "inGain";
    static constexpr char const* outGain = "outGain";

    // filter
    static constexpr char const* filterEnable = "filterEnable";
    static constexpr char const* filterType   = "filterType";
};

auto createParameters() -> juce::AudioProcessorValueTreeState::ParameterLayout;

}  // namespace mc
