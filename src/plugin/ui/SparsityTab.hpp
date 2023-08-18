#pragma once

#include <neo/container/mdspan.hpp>

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_gui_extra/juce_gui_extra.h>

#include <complex>

namespace neo {

struct SparsityTab final
    : juce::Component
    , juce::Value::Listener
{
    explicit SparsityTab(juce::AudioFormatManager& formatManager);
    ~SparsityTab() override = default;

    auto setImpulseResponseFile(juce::File const& file) -> void;

    auto paint(juce::Graphics& g) -> void override;
    auto resized() -> void override;
    auto valueChanged(juce::Value& value) -> void override;

private:
    auto runBenchmarks() -> void;
    auto runWeightingTests() -> void;
    auto runJuceConvolutionBenchmark() -> void;
    auto runDenseConvolutionBenchmark() -> void;
    auto runDenseConvolverBenchmark() -> void;
    auto runSparseConvolverBenchmark() -> void;
    auto updateImages() -> void;

    juce::AudioFormatManager& _formatManager;

    juce::TextButton _render{"Render"};
    juce::PropertyPanel _propertyPanel{};
    juce::TextEditor _fileInfo{};
    juce::ImageComponent _spectogramImage{};
    juce::ImageComponent _histogramImage{};

    juce::File _signalFile{};
    juce::File _filterFile{};
    juce::AudioBuffer<float> _signal{};
    juce::AudioBuffer<float> _filter{};
    stdex::mdarray<std::complex<float>, stdex::dextents<size_t, 2>> _spectrum;

    juce::Value _skip{juce::var{0}};
    juce::Value _dynamicRange{juce::var{90.0F}};
    juce::Value _weighting{juce::var{"A-Weighting"}};
    juce::Value _engine{juce::Array<juce::var>{juce::var{"dense"}}};
};

}  // namespace neo
