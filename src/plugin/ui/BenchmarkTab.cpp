#include "BenchmarkTab.hpp"

#include "core/StringArray.hpp"
#include "dsp/AudioBuffer.hpp"
#include "dsp/AudioFile.hpp"
#include "dsp/DenseConvolution.hpp"
#include "dsp/Spectrum.hpp"

#include <neo/algorithm/peak_normalize.hpp>
#include <neo/container/sparse_matrix.hpp>
#include <neo/fft/transform/fftfreq.hpp>
#include <neo/fft/transform/stft.hpp>
#include <neo/math/a_weighting.hpp>

#include <juce_dsp/juce_dsp.h>

namespace neo {

namespace {

struct JuceConvolver
{
    explicit JuceConvolver(juce::File impulse, juce::dsp::ProcessSpec const& spec)
        : _filter{std::move(impulse)}
        , _spec{spec}
    {
        auto const trim      = juce::dsp::Convolution::Trim::no;
        auto const stereo    = juce::dsp::Convolution::Stereo::yes;
        auto const normalize = juce::dsp::Convolution::Normalise::no;

        _convolver.prepare(spec);
        _convolver.loadImpulseResponse(_filter, stereo, trim, 0, normalize);

        // impulse is loaded on background thread, may not have loaded fast enough in
        // unit-tests
        std::this_thread::sleep_for(std::chrono::milliseconds{2000});
    }

    auto prepare(juce::dsp::ProcessSpec const& spec) -> void { jassertquiet(_spec == spec); }

    auto reset() -> void { _convolver.reset(); }

    template<typename Context>
    auto process(Context const& context) -> void
    {
        _convolver.process(context);
    }

private:
    juce::File _filter;
    juce::dsp::ConvolutionMessageQueue _queue;
    juce::dsp::Convolution _convolver{juce::dsp::Convolution::Latency{0}, _queue};
    juce::dsp::ProcessSpec _spec;
};

}  // namespace

BenchmarkTab::BenchmarkTab(juce::AudioFormatManager& formatManager) : _formatManager{formatManager}
{
    _skip.addListener(this);
    _dynamicRange.addListener(this);
    _weighting.addListener(this);

    _selectSignalFile.onClick = [this] { selectSignalFile(); };
    _render.onClick           = [this] { runBenchmarks(); };

    _fileInfo.setReadOnly(true);
    _fileInfo.setMultiLine(true);
    _spectogramImage.setImagePlacement(juce::RectanglePlacement::centred);
    _histogramImage.setImagePlacement(juce::RectanglePlacement::centred);

    auto const weights     = juce::Array<juce::var>{juce::var{"No Weighting"}, juce::var{"A-Weighting"}};
    auto const weightNames = toStringArray(weights);

    auto const engines = juce::Array<juce::var>{
        juce::var{"juce::Convolution"},
        juce::var{"DenseConvolution"},
        juce::var{"upols_convolver"},
        juce::var{"sparse_upols_convolver"},
    };
    auto const engineNames = toStringArray(engines);

    _propertyPanel.addSection(
        "Sparsity",
        juce::Array<juce::PropertyComponent*>{
            std::make_unique<juce::SliderPropertyComponent>(_skip, "Skip", 0.0, 16.0, 1.0).release(),
            std::make_unique<juce::SliderPropertyComponent>(_dynamicRange, "Dynamic Range", 10.0, 100.0, 0.5).release(),
            std::make_unique<juce::ChoicePropertyComponent>(_weighting, "Weighting", weightNames, weights).release(),
        }
    );

    _propertyPanel.addSection(
        "Render",
        juce::Array<juce::PropertyComponent*>{
            std::make_unique<juce::MultiChoicePropertyComponent>(_engine, "Engine", engineNames, engines).release(),
        }
    );

    addAndMakeVisible(_selectSignalFile);
    addAndMakeVisible(_render);
    addAndMakeVisible(_propertyPanel);
    addAndMakeVisible(_fileInfo);
    addAndMakeVisible(_spectogramImage);
    addAndMakeVisible(_histogramImage);
}

BenchmarkTab::~BenchmarkTab() { _threadPool.removeAllJobs(true, 2000); }

auto BenchmarkTab::setImpulseResponseFile(juce::File const& file) -> void
{
    _filter     = loadAndResample(_formatManager, file, 44'100.0);
    _filterFile = file;

    auto const filterMatrix = to_mdarray(_filter);
    _spectrum               = neo::fft::stft(filterMatrix.to_mdspan(), 1024);

    _fileInfo.setText(file.getFileName() + " (" + juce::String(_filter.getNumSamples()) + ")\n");
    updateImages();

    repaint();
}

auto BenchmarkTab::paint(juce::Graphics& g) -> void { juce::ignoreUnused(g); }

auto BenchmarkTab::resized() -> void
{
    auto bounds = getLocalBounds();

    _fileInfo.setBounds(bounds.removeFromBottom(bounds.proportionOfHeight(0.175)));

    auto right              = bounds.removeFromRight(bounds.proportionOfWidth(0.25));
    auto const buttonHeight = right.proportionOfHeight(0.1);
    _render.setBounds(right.removeFromBottom(buttonHeight).reduced(0, 4));
    _selectSignalFile.setBounds(right.removeFromBottom(buttonHeight).reduced(0, 4));
    _propertyPanel.setBounds(right);

    _spectogramImage.setBounds(bounds.removeFromTop(bounds.proportionOfHeight(0.5)).reduced(4));
    _histogramImage.setBounds(bounds.reduced(4));
}

auto BenchmarkTab::valueChanged(juce::Value& /*value*/) -> void { updateImages(); }

auto BenchmarkTab::selectSignalFile() -> void
{
    auto const* msg         = "Please select a signal file";
    auto const homeDir      = juce::File::getSpecialLocation(juce::File::userMusicDirectory);
    auto const chooserFlags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;
    auto const load         = [this](juce::FileChooser const& chooser) {
        if (chooser.getResults().isEmpty()) {
            return;
        }
        loadSignalFile(chooser.getResult());
    };

    _fileChooser = std::make_unique<juce::FileChooser>(msg, homeDir, _formatManager.getWildcardForAllFormats());
    _fileChooser->launchAsync(chooserFlags, load);
}

auto BenchmarkTab::loadSignalFile(juce::File const& file) -> void
{
    _signalFile = file;
    _signal     = loadAndResample(_formatManager, _signalFile, 44'100.0);
}

auto BenchmarkTab::runBenchmarks() -> void
{
    auto hasEngineEnabled = [this](auto name) {
        if (auto const* array = _engine.getValue().getArray(); array != nullptr) {
            for (auto const& val : *array) {
                if (val.toString() == name) {
                    return true;
                }
            }
        }
        return false;
    };

    if (_signal.getNumChannels() == 0 or _signal.getNumSamples() == 0) {
        return;
    }

    if (hasEngineEnabled("juce::Convolution")) {
        _threadPool.addJob([this] { runJuceConvolutionBenchmark(); });
    }
    if (hasEngineEnabled("DenseConvolution")) {
        _threadPool.addJob([this] { runDenseConvolutionBenchmark(); });
    }
    if (hasEngineEnabled("upols_convolver")) {
        _threadPool.addJob([this] { runDenseConvolverBenchmark(); });
    }
    if (hasEngineEnabled("sparse_upols_convolver")) {
        _threadPool.addJob([this] { runSparseConvolverBenchmark(); });
    }
}

auto BenchmarkTab::runWeightingTests() -> void
{
    auto normalized = _filter;
    juce_normalization(normalized);

    auto const blockSize  = 512ULL;
    auto const skip       = static_cast<std::size_t>(static_cast<int>(_skip.getValue()));
    auto const impulse    = to_mdarray(normalized);
    auto const partitions = neo::fft::uniform_partition(
        stdex::submdspan(impulse.to_mdspan(), stdex::full_extent, std::tuple{blockSize * skip, impulse.extent(1)}),
        blockSize
    );

    auto const scale = [filter = partitions.to_mdspan()] {
        auto max = 0.0F;
        for (auto ch{0U}; ch < filter.extent(0); ++ch) {
            for (auto f{0U}; f < filter.extent(1); ++f) {
                for (auto b{0U}; b < filter.extent(2); ++b) {
                    auto const bin   = std::abs(filter(ch, f, b));
                    auto const power = bin * bin;

                    max = std::max(max, power);
                }
            }
        }
        return 1.0F / max;
    }();

    auto const weighting = [=, this](std::size_t binIndex) {
        if (_weighting.getValue() == "A-Weighting") {
            auto const frequency = neo::fftfreq<float>(blockSize * 2ULL, binIndex, 44'100.0);
            if (juce::exactlyEqual(frequency, 0.0F)) {
                return 0.0F;
            }
            auto const weight = neo::a_weighting(frequency);
            return weight;
        }
        return 0.0F;
    };

    auto count           = 0;
    auto const threshold = -static_cast<float>(_dynamicRange.getValue());

    for (auto f{0U}; f < partitions.extent(1); ++f) {
        for (auto b{0U}; b < partitions.extent(2); ++b) {
            auto const weight = weighting(b);
            for (auto ch{0U}; ch < partitions.extent(0); ++ch) {
                auto const bin   = std::abs(partitions(ch, f, b));
                auto const power = bin * bin;
                auto const dB    = juce::Decibels::gainToDecibels(power * scale, -144.0F) + weight;
                count += static_cast<int>(dB > threshold);
            }
        }
    }

    auto const total = static_cast<double>(partitions.size());
    auto const line  = juce::String(static_cast<double>(count) / total * 100.0, 2) + "% " + juce::String(threshold, 1)
                    + "dB " + _weighting.getValue().toString() + "\n";

    _fileInfo.moveCaretToEnd(false);
    _fileInfo.insertTextAtCaret(line);
}

auto BenchmarkTab::runJuceConvolutionBenchmark() -> void
{
    auto proc = JuceConvolver{
        _filterFile,
        {44'100.0, 512, 2}
    };

    auto start = std::chrono::system_clock::now();
    auto out   = juce::AudioBuffer<float>{_signal.getNumChannels(), _signal.getNumSamples()};
    auto file  = getBenchmarkResultsDirectory().getNonexistentChildFile("jconv", ".wav");

    processBlocks(proc, _signal, out, 512, 44'100.0);

    auto output = to_mdarray(out);
    neo::peak_normalize(output.to_mdspan());

    auto const end     = std::chrono::system_clock::now();
    auto const elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    juce::MessageManager::callAsync([this, elapsed] {
        _fileInfo.moveCaretToEnd(false);
        _fileInfo.insertTextAtCaret("JCONV-DENSE: " + juce::String{elapsed.count()} + "\n");
    });

    writeToWavFile(file, output, 44'100.0, 32);
}

auto BenchmarkTab::runDenseConvolutionBenchmark() -> void
{
    auto out       = juce::AudioBuffer<float>{_signal.getNumChannels(), _signal.getNumSamples()};
    auto inBuffer  = juce::dsp::AudioBlock<float const>{_signal};
    auto outBuffer = juce::dsp::AudioBlock<float>{out};

    auto proc = DenseConvolution{512};
    proc.loadImpulseResponse(_filterFile.createInputStream());
    proc.prepare(juce::dsp::ProcessSpec{
        44'100.0,
        static_cast<std::uint32_t>(512),
        static_cast<std::uint32_t>(inBuffer.getNumChannels()),
    });

    auto start = std::chrono::system_clock::now();
    auto file  = getBenchmarkResultsDirectory().getNonexistentChildFile("tconv2", ".wav");

    for (size_t i{0}; i < outBuffer.getNumSamples(); i += size_t(512)) {
        auto const numSamples = std::min(outBuffer.getNumSamples() - i, size_t(512));

        auto inBlock  = inBuffer.getSubBlock(i, numSamples);
        auto outBlock = outBuffer.getSubBlock(i, numSamples);
        outBlock.copyFrom(inBlock);

        auto ctx = juce::dsp::ProcessContextReplacing<float>{outBlock};
        proc.process(ctx);
    }

    // proc.reset();

    // auto output = to_mdarray(out);
    // neo::peak_normalize(output.to_mdspan());

    auto const end     = std::chrono::system_clock::now();
    auto const elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    juce::MessageManager::callAsync([this, elapsed] {
        _fileInfo.moveCaretToEnd(false);
        _fileInfo.insertTextAtCaret("TCONV2-DENSE: " + juce::String{elapsed.count()} + "\n");
    });

    writeToWavFile(file, to_mdarray(out), 44'100.0, 32);
}

auto BenchmarkTab::runDenseConvolverBenchmark() -> void
{
    auto start = std::chrono::system_clock::now();

    auto output = to_mdarray(neo::dense_convolve(_signal, _filter));
    auto file   = getBenchmarkResultsDirectory().getNonexistentChildFile("tconv_dense", ".wav");

    neo::peak_normalize(output.to_mdspan());

    auto const end     = std::chrono::system_clock::now();
    auto const elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    juce::MessageManager::callAsync([this, elapsed] {
        _fileInfo.moveCaretToEnd(false);
        _fileInfo.insertTextAtCaret("TCONV-DENSE: " + juce::String{elapsed.count()} + "\n");
    });

    writeToWavFile(file, output, 44'100.0, 32);
}

auto BenchmarkTab::runSparseConvolverBenchmark() -> void
{
    auto const start = std::chrono::system_clock::now();

    auto const thresholdDB   = -static_cast<float>(_dynamicRange.getValue());
    auto const thresholdText = juce::String(juce::roundToInt(std::abs(thresholdDB) * 100));

    auto const filename = "tconv_sparse_" + thresholdText;
    auto const file     = getBenchmarkResultsDirectory().getNonexistentChildFile(filename, ".wav");

    auto output = to_mdarray(neo::sparse_convolve(_signal, _filter, thresholdDB));

    neo::peak_normalize(output.to_mdspan());

    auto const end     = std::chrono::system_clock::now();
    auto const elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    juce::MessageManager::callAsync([this, elapsed, thresholdText] {
        auto const line = "TCONV-SPARSE(" + thresholdText + "): " + juce::String{elapsed.count()} + "\n";
        _fileInfo.moveCaretToEnd(false);
        _fileInfo.insertTextAtCaret(line);
    });

    writeToWavFile(file, output, 44'100.0, 32);
}

auto BenchmarkTab::updateImages() -> void
{
    if (_spectrum.size() == 0) {
        return;
    }

    auto const weighting = [this](std::size_t binIndex) {
        if (_weighting.getValue() == "A-Weighting") {
            auto const frequency = neo::fftfreq<float>(1024, binIndex, 44'100.0);
            if (juce::exactlyEqual(frequency, 0.0F)) {
                return 0.0F;
            }
            auto const weight = neo::a_weighting(frequency);
            return weight;
        }
        return 0.0F;
    };

    auto spectogramImage = neo::powerSpectrumImage(_spectrum, weighting, -static_cast<float>(_dynamicRange.getValue()));
    auto histogramImage  = neo::powerHistogramImage(_spectrum, weighting);

    _spectogramImage.setImage(spectogramImage);
    _histogramImage.setImage(histogramImage);

    runWeightingTests();
    repaint();
}

auto BenchmarkTab::getBenchmarkResultsDirectory() -> juce::File
{
    auto directory = juce::File::getSpecialLocation(juce::File::userMusicDirectory)
                         .getChildFile("Perceputual Convolution")
                         .getChildFile("Benchmarks");
    if (not directory.exists()) {
        if (auto result = directory.createDirectory(); result.failed()) {
            DBG(result.getErrorMessage());
            return {};
        }
    }

    return directory;
}
}  // namespace neo
