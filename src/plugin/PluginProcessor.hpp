
#pragma once

#include "PerceptualConvolution.hpp"

#include <juce_audio_processors/juce_audio_processors.h>

namespace neo {

struct ProcessSpecListener
{
    ProcessSpecListener()          = default;
    virtual ~ProcessSpecListener() = default;

    virtual auto processSpecChanged(juce::dsp::ProcessSpec const& spec) -> void = 0;
};

struct PluginProcessor final : juce::AudioProcessor
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

    [[nodiscard]] auto getProcessSpec() const noexcept -> juce::dsp::ProcessSpec;

    auto addProcessSpecListener(ProcessSpecListener* listener) -> void;
    auto removeProcessSpecListener(ProcessSpecListener* listener) -> void;

    [[nodiscard]] auto getState() noexcept -> juce::AudioProcessorValueTreeState&;
    [[nodiscard]] auto getState() const noexcept -> juce::AudioProcessorValueTreeState const&;

private:
    juce::UndoManager _undoManager{};
    juce::AudioProcessorValueTreeState _valueTree;
    std::optional<juce::dsp::ProcessSpec> _spec{std::nullopt};

    PerceptualConvolution _convolution;

    juce::ListenerList<ProcessSpecListener> _specListeners;
};

}  // namespace neo
