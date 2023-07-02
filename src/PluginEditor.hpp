#pragma once

#include <juce_gui_extra/juce_gui_extra.h>

#include "PluginProcessor.hpp"

struct PluginEditor final : juce::AudioProcessorEditor
{
    explicit PluginEditor(PluginProcessor& p);
    ~PluginEditor() noexcept override;

    auto paint(juce::Graphics& g) -> void override;
    auto resized() -> void override;

private:
    juce::SharedResourcePointer<juce::TooltipWindow> _tooltipWindow;
};
