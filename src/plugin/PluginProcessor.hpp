
#pragma once

#include "dsp/ConstantOverlapAdd.hpp"
#include "dsp/DenseConvolution.hpp"

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_extra/juce_gui_extra.h>

namespace neo {
struct PluginProcessor final
    : juce::AudioProcessor
    , juce::AudioProcessorValueTreeState::Listener
{
    PluginProcessor();
    ~PluginProcessor() override;

    auto getName() const -> const juce::String override;

    auto prepareToPlay(double sampleRate, int samplesPerBlock) -> void override;
    auto processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) -> void override;
    using juce::AudioProcessor::processBlock;
    auto releaseResources() -> void override;

    auto isBusesLayoutSupported(BusesLayout const& layouts) const -> bool override;

    auto createEditor() -> juce::AudioProcessorEditor* override;
    auto hasEditor() const -> bool override;

    auto acceptsMidi() const -> bool override;
    auto producesMidi() const -> bool override;
    auto isMidiEffect() const -> bool override;
    auto getTailLengthSeconds() const -> double override;

    auto getNumPrograms() -> int override;
    auto getCurrentProgram() -> int override;
    void setCurrentProgram(int index) override;
    auto getProgramName(int index) -> const juce::String override;
    void changeProgramName(int index, juce::String const& newName) override;

    auto getStateInformation(juce::MemoryBlock& destData) -> void override;
    auto setStateInformation(void const* data, int sizeInBytes) -> void override;

    auto parameterChanged(juce::String const& parameterID, float newValue) -> void override;

    auto getState() noexcept -> juce::AudioProcessorValueTreeState&;
    auto getState() const noexcept -> juce::AudioProcessorValueTreeState const&;

private:
    juce::UndoManager _undoManager{};
    juce::AudioProcessorValueTreeState _valueTree;

    std::unique_ptr<ConstantOverlapAdd<DenseConvolution>> _convolution;
    // juce::AudioParameterFloat& _inGain;
    // juce::AudioParameterFloat& _outGain;
};

}  // namespace neo
