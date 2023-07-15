#include "PluginEditor.hpp"

#include "dsp/convolution.hpp"
#include "dsp/normalize.hpp"
#include "dsp/render.hpp"
#include "dsp/resample.hpp"
#include "dsp/spectogram.hpp"
#include "dsp/stft.hpp"
#include "dsp/wav.hpp"
#include "neo/convolution/container/sparse_matrix.hpp"

#include <span>

namespace neo {

namespace {

struct JuceConvolver
{
    explicit JuceConvolver(juce::File impulse, juce::dsp::ProcessSpec const& spec)
        : _impulse{std::move(impulse)}
        , _spec{spec}
    {
        auto const trim      = juce::dsp::Convolution::Trim::no;
        auto const stereo    = juce::dsp::Convolution::Stereo::yes;
        auto const normalize = juce::dsp::Convolution::Normalise::no;

        _convolver.prepare(spec);
        _convolver.loadImpulseResponse(_impulse, stereo, trim, 0, normalize);

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
    juce::File _impulse;
    juce::dsp::ConvolutionMessageQueue _queue;
    juce::dsp::Convolution _convolver{juce::dsp::Convolution::Latency{0}, _queue};
    juce::dsp::ProcessSpec _spec;
};

}  // namespace

PluginEditor::PluginEditor(PluginProcessor& p) : AudioProcessorEditor(&p)
{
    _formats.registerBasicFormats();

    _openFile.onClick      = [this] { openFile(); };
    _runBenchmarks.onClick = [this] { runBenchmarks(); };

    _threshold.setRange({-144.0, -10.0}, 0.0);
    _threshold.setValue(-90.0, juce::dontSendNotification);
    _threshold.onDragEnd = [this] {
        auto img = fft::powerSpectrumImage(_spectrum, static_cast<float>(_threshold.getValue()));
        _spectogramImage.setImage(img);
    };

    _fileInfo.setReadOnly(true);
    _fileInfo.setMultiLine(true);
    _spectogramImage.setImagePlacement(juce::RectanglePlacement::centred);
    _histogramImage.setImagePlacement(juce::RectanglePlacement::centred);
    _tooltipWindow->setMillisecondsBeforeTipAppears(750);

    addAndMakeVisible(_openFile);
    addAndMakeVisible(_runBenchmarks);
    addAndMakeVisible(_threshold);
    addAndMakeVisible(_fileInfo);
    addAndMakeVisible(_spectogramImage);
    addAndMakeVisible(_histogramImage);

    setResizable(true, true);
    setSize(600, 400);
}

PluginEditor::~PluginEditor() noexcept { setLookAndFeel(nullptr); }

auto PluginEditor::paint(juce::Graphics& g) -> void
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

auto PluginEditor::resized() -> void
{
    auto bounds = getLocalBounds();

    auto controls = bounds.removeFromTop(bounds.proportionOfHeight(0.1));
    auto width    = controls.proportionOfWidth(0.333);
    _openFile.setBounds(controls.removeFromLeft(width));
    _runBenchmarks.setBounds(controls.removeFromLeft(width));
    _threshold.setBounds(controls);

    _fileInfo.setBounds(bounds.removeFromLeft(bounds.proportionOfWidth(0.15)));
    _spectogramImage.setBounds(bounds.removeFromLeft(bounds.proportionOfWidth(0.60)));
    _histogramImage.setBounds(bounds.reduced(4));
}

auto PluginEditor::openFile() -> void
{
    auto const* msg         = "Please select a impulse response";
    auto const homeDir      = juce::File::getSpecialLocation(juce::File::userMusicDirectory);
    auto const chooserFlags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;
    auto const load         = [this](juce::FileChooser const& chooser) {
        if (chooser.getResults().isEmpty()) { return; }

        auto const file     = chooser.getResult();
        auto const filename = file.getFileNameWithoutExtension();

        _impulse  = loadAndResample(_formats, file, 44'100.0);
        _spectrum = fft::stft(_impulse, 1024);

        _spectogramImage.setImage(fft::powerSpectrumImage(_spectrum, static_cast<float>(_threshold.getValue())));
        _histogramImage.setImage(fft::powerHistogramImage(_spectrum));

        _fileInfo.setText(filename + " (" + juce::String(_impulse.getNumSamples()) + ")");

        repaint();
    };

    _fileChooser = std::make_unique<juce::FileChooser>(msg, homeDir, _formats.getWildcardForAllFormats());
    _fileChooser->launchAsync(chooserFlags, load);
}

auto PluginEditor::runBenchmarks() -> void
{
    _signalFile = juce::File{R"(C:\Users\tobias\Music\Loops\Drums.wav)"};
    _filterFile = juce::File{R"(C:\Users\tobias\Music\Samples\IR\LexiconPCM90 Halls\ORCH_gothic hall.WAV)"};

    _signal = loadAndResample(_formats, _signalFile, 44'100.0);
    _filter = loadAndResample(_formats, _filterFile, 44'100.0);

    runDynamicRangeTests();
    runJuceConvolverBenchmark();
    runDenseConvolverBenchmark();
    runDenseStereoConvolverBenchmark();
    runSparseConvolverBenchmark();
}

auto PluginEditor::runDynamicRangeTests() -> void
{
    auto N          = 1024;
    auto normalized = _signal;
    juce_normalization(normalized);

    auto coeffs = neo::fft::stft(normalized, N);

    for (auto frame{0U}; frame < coeffs.extent(0); ++frame) {
        auto mins = std::array<float, 2>{999.0F, 999.0F};
        auto maxs = std::array<float, 2>{0.0F, 0.0F};

        for (auto bin{0U}; bin < coeffs.extent(1); ++bin) {
            auto const coeff = coeffs(frame, bin);

            mins[0] = std::min(mins[0], coeff.real());
            maxs[0] = std::max(maxs[0], coeff.real());

            mins[1] = std::min(mins[1], coeff.imag());
            maxs[1] = std::max(maxs[1], coeff.imag());
        }

        auto const realRange = std::abs(maxs[0] - mins[0]);
        auto const imagRange = std::abs(maxs[1] - mins[1]);
        auto const range     = std::max(realRange, imagRange);
        auto const scale     = 1.0F / range;

        std::printf("frame: %-3u range: %.6f scale: %.6f\n", frame, range, scale);
    }
}

auto PluginEditor::runJuceConvolverBenchmark() -> void
{
    auto proc = JuceConvolver{
        _filterFile,
        {44'100.0, 512, 2}
    };

    auto start  = std::chrono::system_clock::now();
    auto output = juce::AudioBuffer<float>{_signal.getNumChannels(), _signal.getNumSamples()};
    auto file   = juce::File{R"(C:\Users\tobias\Music)"}.getNonexistentChildFile("jconv", ".wav");

    processBlocks(proc, _signal, output, 512, 44'100.0);
    peak_normalization(std::span{output.getWritePointer(0), size_t(output.getNumSamples())});
    peak_normalization(std::span{output.getWritePointer(1), size_t(output.getNumSamples())});

    auto end = std::chrono::system_clock::now();
    std::cout << "JCONV: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n';

    writeToWavFile(file, output, 44'100.0, 32);
}

auto PluginEditor::runDenseConvolverBenchmark() -> void
{
    auto start = std::chrono::system_clock::now();

    auto output = fft::dense_convolve(_signal, _filter);
    auto file   = juce::File{R"(C:\Users\tobias\Music)"}.getNonexistentChildFile("tconv_dense", ".wav");

    peak_normalization(std::span{output.getWritePointer(0), size_t(output.getNumSamples())});
    peak_normalization(std::span{output.getWritePointer(1), size_t(output.getNumSamples())});

    auto end = std::chrono::system_clock::now();
    std::cout << "TCONV-DENSE: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n';

    writeToWavFile(file, output, 44'100.0, 32);
}

auto PluginEditor::runDenseStereoConvolverBenchmark() -> void
{
    auto start = std::chrono::system_clock::now();

    auto output = fft::dense_stereo_convolve(_signal, _filter);
    auto file   = juce::File{R"(C:\Users\tobias\Music)"}.getNonexistentChildFile("tconv_dense_stereo", ".wav");

    peak_normalization(std::span{output.getWritePointer(0), size_t(output.getNumSamples())});
    peak_normalization(std::span{output.getWritePointer(1), size_t(output.getNumSamples())});

    auto end = std::chrono::system_clock::now();
    std::cout << "TCONV-DENSE-STEREO: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << '\n';

    writeToWavFile(file, output, 44'100.0, 32);
}

auto PluginEditor::runSparseConvolverBenchmark() -> void
{
    auto start = std::chrono::system_clock::now();

    auto thresholdDB = -40.0F;
    auto output      = fft::sparse_convolve(_signal, _filter, thresholdDB);
    auto file        = juce::File{R"(C:\Users\tobias\Music)"}.getNonexistentChildFile("tconv_sparse_40", ".wav");

    peak_normalization(std::span{output.getWritePointer(0), size_t(output.getNumSamples())});
    peak_normalization(std::span{output.getWritePointer(1), size_t(output.getNumSamples())});

    auto end = std::chrono::system_clock::now();
    std::cout << "TCONV-SPARSE(40): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << '\n';

    writeToWavFile(file, output, 44'100.0, 32);
}

}  // namespace neo
