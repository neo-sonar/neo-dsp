#pragma once

#include "PluginProcessor.hpp"

#include "neo/convolution/container/mdspan.hpp"

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
    juce::AudioBuffer<float> _impulse;
    KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>> _spectrum;

    juce::TextButton _openFile{"Open File"};
    juce::Slider _threshold{juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight};
    juce::TextEditor _fileInfo{};
    juce::ImageComponent _spectogramImage{};
    juce::ImageComponent _histogramImage{};

    std::unique_ptr<juce::FileChooser> _fileChooser{nullptr};
    juce::SharedResourcePointer<juce::TooltipWindow> _tooltipWindow;
};

}  // namespace neo
