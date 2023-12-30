// SPDX-License-Identifier: MIT

#include "PluginParameter.hpp"

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

namespace neo {

template<typename ParamT, typename StringLike, typename... Args>
auto makeParameter(StringLike&& id, Args&&... args)
{
    return std::make_unique<ParamT>(juce::ParameterID{std::forward<StringLike>(id), 1}, std::forward<Args>(args)...);
}

template<typename StringLike, typename... Args>
auto makeFloatParameter(StringLike&& id, Args&&... args)
{
    return makeParameter<juce::AudioParameterFloat>(std::forward<StringLike>(id), std::forward<Args>(args)...);
}

template<typename StringLike, typename... Args>
auto makeIntParameter(StringLike&& id, Args&&... args)
{
    return makeParameter<juce::AudioParameterInt>(std::forward<StringLike>(id), std::forward<Args>(args)...);
}

template<typename StringLike, typename... Args>
auto makeChoiceParameter(StringLike&& id, Args&&... args)
{
    return makeParameter<juce::AudioParameterChoice>(std::forward<StringLike>(id), std::forward<Args>(args)...);
}

template<typename StringLike, typename... Args>
auto makeBoolParameter(StringLike&& id, Args&&... args)
{
    return makeParameter<juce::AudioParameterBool>(std::forward<StringLike>(id), std::forward<Args>(args)...);
}

auto createParameters() -> juce::AudioProcessorValueTreeState::ParameterLayout
{
    auto const normalised = juce::NormalisableRange{0.0F, 1.0F};
    auto const gainRange  = [] {
        auto range = juce::NormalisableRange{0.0F, 4.0F};
        range.setSkewForCentre(1.0F);
        return range;
    }();

    return {
        makeFloatParameter(ParamID::inGain, "Input Gain", gainRange, 1.0F),
        makeFloatParameter(ParamID::wet, "Wet", normalised, 1.0F),
        makeFloatParameter(ParamID::outGain, "Output Gain", gainRange, 1.0F),
    };
}

namespace {
template<typename T>
auto getParameter(juce::AudioProcessorValueTreeState& vts, juce::StringRef id) -> T
{
    auto* raw       = vts.getParameter(id);
    auto* parameter = dynamic_cast<T>(raw);
    return parameter;
}
}  // namespace

auto getFloatParameter(juce::AudioProcessorValueTreeState& vts, juce::StringRef id) -> juce::AudioParameterFloat*
{
    return getParameter<juce::AudioParameterFloat*>(vts, id);
}

auto getChoiceParameter(juce::AudioProcessorValueTreeState& vts, juce::StringRef id) -> juce::AudioParameterChoice*
{
    return getParameter<juce::AudioParameterChoice*>(vts, id);
}

auto getIntParameter(juce::AudioProcessorValueTreeState& vts, juce::StringRef id) -> juce::AudioParameterInt*
{
    return getParameter<juce::AudioParameterInt*>(vts, id);
}

auto getBoolParameter(juce::AudioProcessorValueTreeState& vts, juce::StringRef id) -> juce::AudioParameterBool*
{
    return getParameter<juce::AudioParameterBool*>(vts, id);
}

}  // namespace neo
