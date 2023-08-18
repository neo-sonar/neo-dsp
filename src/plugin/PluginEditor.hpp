#pragma once

#include "PluginProcessor.hpp"
#include "ui/ParameterTab.hpp"
#include "ui/SparsityTab.hpp"

#include <neo/container/mdspan.hpp>

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_gui_extra/juce_gui_extra.h>

namespace neo {

struct PluginEditor final : juce::AudioProcessorEditor
{
    explicit PluginEditor(PluginProcessor& p);
    ~PluginEditor() noexcept override;

    auto paint(juce::Graphics& g) -> void override;
    auto resized() -> void override;

private:
    ParameterTab _parameter;
    SparsityTab _sparsity;
    juce::TabbedComponent _tabs;

    juce::SharedResourcePointer<juce::TooltipWindow> _tooltipWindow;
};

}  // namespace neo
