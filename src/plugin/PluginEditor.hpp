#pragma once

#include "neo/fft/container/mdspan.hpp"
#include "PluginProcessor.hpp"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_gui_extra/juce_gui_extra.h>

namespace neo {

struct PluginEditor final
    : juce::AudioProcessorEditor
    , juce::Value::Listener
{
    explicit PluginEditor(PluginProcessor& p);
    ~PluginEditor() noexcept override;

    auto paint(juce::Graphics& g) -> void override;
    auto resized() -> void override;
    auto valueChanged(juce::Value& value) -> void override;

private:
    auto openFile() -> void;
    auto runBenchmarks() -> void;
    auto runWeightingTests() -> void;
    auto runDynamicRangeTests() -> void;
    auto runJuceConvolverBenchmark() -> void;
    auto runDenseConvolverBenchmark() -> void;
    auto runDenseStereoConvolverBenchmark() -> void;
    auto runSparseConvolverBenchmark() -> void;
    auto updateImages() -> void;

    juce::AudioFormatManager _formats;

    juce::TextButton _openFile{"Open Impulse"};
    juce::TextButton _render{"Render"};
    juce::PropertyPanel _propertyPanel{};
    juce::TextEditor _fileInfo{};
    juce::ImageComponent _spectogramImage{};
    juce::ImageComponent _histogramImage{};

    juce::File _signalFile{};
    juce::File _filterFile{};
    juce::AudioBuffer<float> _signal{};
    juce::AudioBuffer<float> _filter{};
    KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>> _spectrum;

    juce::Value _dynamicRange{juce::var{90.0F}};
    juce::Value _weighting{juce::var{"A-Weighting"}};
    juce::Value _engine{juce::Array<juce::var>{juce::var{"dense"}}};

    std::unique_ptr<juce::FileChooser> _fileChooser{nullptr};
    juce::SharedResourcePointer<juce::TooltipWindow> _tooltipWindow;
};

}  // namespace neo
