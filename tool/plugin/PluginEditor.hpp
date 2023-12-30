// SPDX-License-Identifier: MIT

#pragma once

#include "PluginProcessor.hpp"
#include "ui/BenchmarkTab.hpp"
#include "ui/ParameterTab.hpp"

#include <neo/container/mdspan.hpp>

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_gui_extra/juce_gui_extra.h>

namespace neo {

struct PluginEditor final
    : juce::AudioProcessorEditor
    , ProcessSpecListener
{
    explicit PluginEditor(PluginProcessor& p);
    ~PluginEditor() noexcept override;

    auto paint(juce::Graphics& g) -> void override;
    auto resized() -> void override;
    auto processSpecChanged(juce::dsp::ProcessSpec const& spec) -> void override;

private:
    auto openFile() -> void;

    juce::AudioFormatManager _formats;

    juce::TextButton _openFile{"Open Impulse"};

    ParameterTab _parameterTab;
    BenchmarkTab _benchmarkTab;
    juce::TabbedComponent _tabs;
    juce::Label _sampleRateLabel{"Samplerate: ??"};
    juce::Label _blockSizeLabel{"Block-size: ??"};
    juce::Label _numChannelsLabel{"Channels: ??"};

    std::unique_ptr<juce::FileChooser> _fileChooser{nullptr};
    juce::SharedResourcePointer<juce::TooltipWindow> _tooltipWindow;
};

}  // namespace neo
