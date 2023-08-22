#pragma once

#include "dsp/DenseConvolution.hpp"

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

namespace neo {

struct PerceptualConvolution
{
    explicit PerceptualConvolution(juce::AudioProcessorValueTreeState& apvts);

    auto prepare(juce::dsp::ProcessSpec const& spec) -> void;
    auto process(juce::dsp::ProcessContextReplacing<float> const& context) -> void;
    auto reset() -> void;

private:
    std::unique_ptr<DenseConvolution> _convolution;
    juce::dsp::DryWetMixer<float> _mixer;

    juce::AudioParameterFloat& _inGain;
    juce::AudioParameterFloat& _outGain;
    juce::AudioParameterFloat& _wet;
};

}  // namespace neo
