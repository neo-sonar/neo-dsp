#pragma once

#include <neo/container/mdspan.hpp>

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_gui_extra/juce_gui_extra.h>

#include <complex>

namespace neo {

struct BenchmarkTab final
    : juce::Component
    , juce::Value::Listener
{
    explicit BenchmarkTab(juce::AudioFormatManager& formatManager);
    ~BenchmarkTab() override;

    auto setImpulseResponseFile(juce::File const& file) -> void;

    auto paint(juce::Graphics& g) -> void override;
    auto resized() -> void override;
    auto valueChanged(juce::Value& value) -> void override;

private:
    auto selectSignalFile() -> void;
    auto loadSignalFile(juce::File const& file) -> void;

    auto runBenchmarks() -> void;
    auto runWeightingTests() -> void;
    auto runJuceConvolutionBenchmark() -> void;
    auto runDenseConvolutionBenchmark() -> void;
    auto runDenseConvolverBenchmark() -> void;
    auto runSparseConvolverBenchmark() -> void;
    auto runSparseQualityTests() -> void;

    auto updateImages() -> void;

    [[nodiscard]] static auto getBenchmarkResultsDirectory() -> juce::File;

    juce::AudioFormatManager& _formatManager;

    juce::TextButton _selectSignalFile{"Select Signal File"};
    juce::TextButton _render{"Render"};
    juce::PropertyPanel _propertyPanel{};
    juce::TextEditor _fileInfo{};
    juce::ImageComponent _spectogramImage{};
    juce::ImageComponent _histogramImage{};

    juce::File _signalFile{};
    juce::File _filterFile{};
    BufferWithSampleRate<float> _signal{};
    BufferWithSampleRate<float> _filter{};

    stdex::mdarray<std::complex<float>, stdex::dextents<size_t, 3>> _spectrum;
    stdex::mdarray<std::complex<float>, stdex::dextents<size_t, 3>> _partitions;

    // Sparsity
    juce::Value _binsToKeep{juce::var{10}};
    juce::Value _dynamicRange{juce::var{90.0F}};
    juce::Value _weighting{juce::var{"A-Weighting"}};
    juce::Value _engine{juce::Array<juce::var>{juce::var{"dense"}}};

    // Quality
    juce::Value _stftWindowSize{juce::var{512}};

    std::unique_ptr<juce::FileChooser> _fileChooser{nullptr};
    juce::ThreadPool _threadPool{juce::ThreadPoolOptions{}.withNumberOfThreads(1).withThreadName("Benchmark Thread")};
};

}  // namespace neo
