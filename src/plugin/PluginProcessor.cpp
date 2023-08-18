
#include "PluginProcessor.hpp"

#include "PluginEditor.hpp"
#include "PluginParameter.hpp"

#include <juce_audio_processors/juce_audio_processors.h>

namespace neo {

PluginProcessor::PluginProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true))
    , _valueTree{*this, nullptr, juce::Identifier("PerceptualConvolution"), mc::createParameters()}
// , _inGain{*mc::getFloatParameter(_valueTree, mc::ParamID::inGain)}
// , _outGain{*mc::getFloatParameter(_valueTree, mc::ParamID::outGain)}
{}

PluginProcessor::~PluginProcessor() = default;

auto PluginProcessor::getName() const -> const juce::String { return JucePlugin_Name; }

auto PluginProcessor::acceptsMidi() const -> bool { return false; }

auto PluginProcessor::producesMidi() const -> bool { return false; }

auto PluginProcessor::isMidiEffect() const -> bool { return false; }

auto PluginProcessor::getTailLengthSeconds() const -> double { return 0.0; }

auto PluginProcessor::getNumPrograms() -> int { return 1; }

auto PluginProcessor::getCurrentProgram() -> int { return 0; }

auto PluginProcessor::setCurrentProgram(int index) -> void { juce::ignoreUnused(index); }

auto PluginProcessor::getProgramName(int index) -> const juce::String
{
    juce::ignoreUnused(index);
    return {};
}

auto PluginProcessor::changeProgramName(int index, juce::String const& newName) -> void
{
    juce::ignoreUnused(index, newName);
}

auto PluginProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) -> void
{
    auto const spec = juce::dsp::ProcessSpec{
        sampleRate,
        static_cast<std::uint32_t>(samplesPerBlock),
        static_cast<std::uint32_t>(getMainBusNumInputChannels()),
    };

    auto impulse = juce::File{R"(C:\Users\tobias\Music\Samples\IR\LexiconPCM90 Halls\ORCH_large hall.WAV)"};

    auto const K = neo::next_power_of_two(spec.maximumBlockSize);
    _convolution = std::make_unique<ConstantOverlapAdd<DenseConvolution>>(neo::ilog2(K), 0);
    _convolution->processor().loadImpulseResponse(impulse.createInputStream());
    _convolution->prepare(spec);
}

auto PluginProcessor::releaseResources() -> void
{
    if (_convolution) {
        _convolution->reset();
    }
}

auto PluginProcessor::isBusesLayoutSupported(BusesLayout const& layouts) const -> bool
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
        && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo()) {
        return false;
    }

    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet()) {
        return false;
    }

    return true;
}

void PluginProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{

    juce::ignoreUnused(midiMessages);

    juce::ScopedNoDenormals const noDenormals;

    for (auto i = getTotalNumInputChannels(); i < getTotalNumOutputChannels(); ++i) {
        buffer.clear(i, 0, buffer.getNumSamples());
    }

    // buffer.applyGain(_inGain);

    if (_convolution) {
        auto block   = juce::dsp::AudioBlock<float>{buffer};
        auto context = juce::dsp::ProcessContextReplacing{block};
        _convolution->process(context);
    }

    // buffer.applyGain(_outGain);
}

auto PluginProcessor::parameterChanged(juce::String const& parameterID, float newValue) -> void
{
    juce::ignoreUnused(newValue, parameterID);
}

auto PluginProcessor::hasEditor() const -> bool { return true; }

auto PluginProcessor::createEditor() -> juce::AudioProcessorEditor* { return new PluginEditor(*this); }

auto PluginProcessor::getStateInformation(juce::MemoryBlock& destData) -> void
{
    juce::MemoryOutputStream stream(destData, false);
    _valueTree.state.writeToStream(stream);
}

auto PluginProcessor::setStateInformation(void const* data, int sizeInBytes) -> void
{
    juce::ValueTree const tree = juce::ValueTree::readFromData(data, static_cast<size_t>(sizeInBytes));
    jassert(tree.isValid());
    if (tree.isValid()) {
        _valueTree.state = tree;
    }
}

auto PluginProcessor::getState() noexcept -> juce::AudioProcessorValueTreeState& { return _valueTree; }

auto PluginProcessor::getState() const noexcept -> juce::AudioProcessorValueTreeState const& { return _valueTree; }

}  // namespace neo

// This creates new instances of the plugin..
auto JUCE_CALLTYPE createPluginFilter() -> juce::AudioProcessor* { return new neo::PluginProcessor{}; }
