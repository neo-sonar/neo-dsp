#pragma once

#include "PluginProcessor.hpp"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_gui_extra/juce_gui_extra.h>

namespace neo
{

struct PluginEditor final : juce::AudioProcessorEditor
{
    explicit PluginEditor(PluginProcessor& p);
    ~PluginEditor() noexcept override;

    auto paint(juce::Graphics& g) -> void override;
    auto resized() -> void override;

private:
    auto openFile() -> void;

    juce::AudioFormatManager _formats;

    juce::TextButton _openFile{"Open File"};
    juce::TextEditor _fileInfo{};
    juce::ImageComponent _image{};

    std::unique_ptr<juce::FileChooser> _fileChooser{nullptr};
    juce::SharedResourcePointer<juce::TooltipWindow> _tooltipWindow;
};

}  // namespace neo
