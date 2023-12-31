// SPDX-License-Identifier: MIT

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_data_structures/juce_data_structures.h>

namespace neo {

struct ParamID
{
    static constexpr char const* inGain  = "inGain";
    static constexpr char const* outGain = "outGain";
    static constexpr char const* wet     = "wet";
};

auto createParameters() -> juce::AudioProcessorValueTreeState::ParameterLayout;

auto getFloatParameter(juce::AudioProcessorValueTreeState& vts, juce::StringRef id) -> juce::AudioParameterFloat*;
auto getChoiceParameter(juce::AudioProcessorValueTreeState& vts, juce::StringRef id) -> juce::AudioParameterChoice*;
auto getIntParameter(juce::AudioProcessorValueTreeState& vts, juce::StringRef id) -> juce::AudioParameterInt*;
auto getBoolParameter(juce::AudioProcessorValueTreeState& vts, juce::StringRef id) -> juce::AudioParameterBool*;

}  // namespace neo
