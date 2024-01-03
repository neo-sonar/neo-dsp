// SPDX-License-Identifier: MIT

#include "PerceptualConvolution.hpp"

#include "PluginParameter.hpp"

namespace neo {

PerceptualConvolution::PerceptualConvolution(juce::AudioProcessorValueTreeState& apvts)
    : _inGain{*getFloatParameter(apvts, ParamID::inGain)}
    , _outGain{*getFloatParameter(apvts, ParamID::outGain)}
    , _wet{*getFloatParameter(apvts, ParamID::wet)}
{}

auto PerceptualConvolution::prepare(juce::dsp::ProcessSpec const& spec) -> void
{
    auto impulse = juce::File{R"(/home/tobante/Music/Samples/4.4s_Dream_Hall.wav)"};

    _convolution = std::make_unique<DenseConvolution>(static_cast<int>(spec.maximumBlockSize));
    _convolution->loadImpulseResponse(impulse.createInputStream());
    _convolution->prepare(spec);
    // setLatencySamples(juce::nextPowerOfTwo(static_cast<int>(spec.maximumBlockSize)));

    _mixer.prepare(spec);
    _mixer.setMixingRule(juce::dsp::DryWetMixingRule::balanced);
    // _mixer.setWetLatency(static_cast<float>(juce::nextPowerOfTwo(static_cast<int>(spec.maximumBlockSize))));
}

auto PerceptualConvolution::process(juce::dsp::ProcessContextReplacing<float> const& context) -> void
{
    auto block = context.getOutputBlock();

    block.multiplyBy(_inGain);

    if (_convolution) {
        _mixer.pushDrySamples(block);

        _convolution->process(context);

        _mixer.setWetMixProportion(_wet);
        _mixer.mixWetSamples(block);
    }

    block.multiplyBy(_outGain);
}

auto PerceptualConvolution::reset() -> void
{
    if (_convolution) {
        _convolution->reset();
    }
}

}  // namespace neo
