// SPDX-License-Identifier: MIT

#pragma once

#include <neo/container/mdspan.hpp>

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_extra/juce_gui_extra.h>

namespace neo {

struct ParameterTab final : juce::Component
{
    explicit ParameterTab(juce::AudioProcessor& processor);
    ~ParameterTab() override = default;

    auto paint(juce::Graphics& g) -> void override;
    auto resized() -> void override;

private:
    juce::GenericAudioProcessorEditor _editor;
};

}  // namespace neo
