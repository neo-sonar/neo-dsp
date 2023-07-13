#include "PluginEditor.hpp"

#include "convolution.hpp"
#include "normalize.hpp"
#include "render.hpp"
#include "resample.hpp"
#include "spectogram.hpp"
#include "stft.hpp"
#include "wav.hpp"

#include "neo/convolution/container/sparse_matrix.hpp"

#include <span>

namespace neo
{

[[maybe_unused]] static auto testSparseMatrix() -> bool
{
    auto lhs = KokkosEx::mdarray<float, Kokkos::dextents<std::size_t, 2>>{16, 32};
    std::fill(lhs.data(), std::next(lhs.data(), std::ssize(lhs)), 1.0F);

    auto rhs = sparse_matrix<float>{16, 32};
    jassertquiet(rhs.rows() == 16);
    jassertquiet(rhs.columns() == 32);

    auto accumulator = std::vector<float>(lhs.extent(1));
    schur_product_accumulate_columnwise(lhs.to_mdspan(), rhs, std::span<float>{accumulator});
    jassert(std::all_of(accumulator.begin(), accumulator.end(), [](auto x) { return x == 0.0F; }));

    rhs.insert(0, 0, 2.0F);
    schur_product_accumulate_columnwise(lhs.to_mdspan(), rhs, std::span<float>{accumulator});
    jassert(accumulator[0] == 2.0F);
    jassert(std::all_of(std::next(accumulator.begin()), accumulator.end(), [](auto x) { return x == 0.0F; }));
    // std::fill(accumulator.begin(), accumulator.end(), 0.0F);

    auto other = sparse_matrix<float>{lhs.to_mdspan(), [](auto v) { return v >= 1.0F; }};
    jassert(other.rows() == lhs.extent(0));
    jassert(other.columns() == lhs.extent(1));
    jassert(other.value_container().size() == lhs.size());

    other = sparse_matrix<float>{lhs.to_mdspan(), [](auto v) { return v >= 2.0F; }};
    jassert(other.rows() == lhs.extent(0));
    jassert(other.columns() == lhs.extent(1));
    jassert(other.value_container().size() == 0);

    auto row = std::vector<float>(lhs.extent(1));
    std::fill(row.begin(), row.end(), 2.0F);
    other.insert_row(0, std::span{row}, [](auto) { return true; });
    jassert(other.value_container().size() == 32);
    jassert(other.column_container().size() == 32);
    return true;
}

PluginEditor::PluginEditor(PluginProcessor& p) : AudioProcessorEditor(&p)
{
    jassert(testSparseMatrix());

    _formats.registerBasicFormats();

    _openFile.onClick = [this] { openFile(); };
    _threshold.setRange({-144.0, -10.0}, 0.0);
    _threshold.setValue(-90.0, juce::dontSendNotification);
    _threshold.onDragEnd = [this]
    {
        auto img = fft::powerSpectrumImage(_spectrum, static_cast<float>(_threshold.getValue()));
        _spectogramImage.setImage(img);
    };

    _fileInfo.setReadOnly(true);
    _fileInfo.setMultiLine(true);
    _spectogramImage.setImagePlacement(juce::RectanglePlacement::centred);
    _histogramImage.setImagePlacement(juce::RectanglePlacement::centred);
    _tooltipWindow->setMillisecondsBeforeTipAppears(750);

    addAndMakeVisible(_openFile);
    addAndMakeVisible(_threshold);
    addAndMakeVisible(_fileInfo);
    addAndMakeVisible(_spectogramImage);
    addAndMakeVisible(_histogramImage);

    setResizable(true, true);
    setSize(600, 400);

    // auto const signalFile = juce::File{R"(C:\Users\tobias\Music\Loops\Drums.wav)"};
    // auto const filterFile = juce::File{R"(C:\Users\tobias\Music\Samples\IR\LexiconPCM90 Halls\ORCH_gothic
    // hall.WAV)"};

    // auto const signal = loadAndResample(_formats, signalFile, 44'100.0);
    // auto const filter = loadAndResample(_formats, filterFile, 44'100.0);

    // {
    //     auto start = std::chrono::system_clock::now();

    //     auto proc   = fft::juce_convolver{filterFile};
    //     auto output = juce::AudioBuffer<float>{signal.getNumChannels(), signal.getNumSamples()};
    //     auto file   = juce::File{R"(C:\Users\tobias\Music)"}.getNonexistentChildFile("jconv", ".wav");

    //     processBlocks(proc, signal, output, 512, 44'100.0);
    //     peak_normalization(std::span{output.getWritePointer(0), size_t(output.getNumSamples())});
    //     peak_normalization(std::span{output.getWritePointer(1), size_t(output.getNumSamples())});

    //     auto end = std::chrono::system_clock::now();
    //     std::cout << "JCONV: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << '\n';

    //     writeToWavFile(file, output, 44'100.0, 32);
    // }

    // {
    //     auto start = std::chrono::system_clock::now();

    //     auto output = fft::convolve(signal, filter, -25.0F);
    //     auto file   = juce::File{R"(C:\Users\tobias\Music)"}.getNonexistentChildFile("tconv_25", ".wav");

    //     peak_normalization(std::span{output.getWritePointer(0), size_t(output.getNumSamples())});
    //     peak_normalization(std::span{output.getWritePointer(1), size_t(output.getNumSamples())});

    //     auto end = std::chrono::system_clock::now();
    //     std::cout << "TCONV(25): " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    //               << '\n';

    //     writeToWavFile(file, output, 44'100.0, 32);
    // }

    // {
    //     auto start = std::chrono::system_clock::now();

    //     auto output = fft::convolve(signal, filter, -30.0F);
    //     auto file   = juce::File{R"(C:\Users\tobias\Music)"}.getNonexistentChildFile("tconv_30", ".wav");

    //     peak_normalization(std::span{output.getWritePointer(0), size_t(output.getNumSamples())});
    //     peak_normalization(std::span{output.getWritePointer(1), size_t(output.getNumSamples())});

    //     auto end = std::chrono::system_clock::now();
    //     std::cout << "TCONV(30): " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    //               << '\n';

    //     writeToWavFile(file, output, 44'100.0, 32);
    // }

    // {
    //     auto start = std::chrono::system_clock::now();

    //     auto output = fft::convolve(signal, filter, -40.0F);
    //     auto file   = juce::File{R"(C:\Users\tobias\Music)"}.getNonexistentChildFile("tconv_40", ".wav");

    //     peak_normalization(std::span{output.getWritePointer(0), size_t(output.getNumSamples())});
    //     peak_normalization(std::span{output.getWritePointer(1), size_t(output.getNumSamples())});

    //     auto end = std::chrono::system_clock::now();
    //     std::cout << "TCONV(40): " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    //               << '\n';

    //     writeToWavFile(file, output, 44'100.0, 32);
    // }

    // {
    //     auto start = std::chrono::system_clock::now();

    //     auto output = fft::convolve(signal, filter, -60.0F);
    //     auto file   = juce::File{R"(C:\Users\tobias\Music)"}.getNonexistentChildFile("tconv_60", ".wav");

    //     peak_normalization(std::span{output.getWritePointer(0), size_t(output.getNumSamples())});
    //     peak_normalization(std::span{output.getWritePointer(1), size_t(output.getNumSamples())});

    //     auto end = std::chrono::system_clock::now();
    //     std::cout << "TCONV(60): " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    //               << '\n';

    //     writeToWavFile(file, output, 44'100.0, 32);
    // }

    // {
    //     auto start = std::chrono::system_clock::now();

    //     auto output = fft::convolve(signal, filter, -80.0F);
    //     auto file   = juce::File{R"(C:\Users\tobias\Music)"}.getNonexistentChildFile("tconv_80", ".wav");

    //     peak_normalization(std::span{output.getWritePointer(0), size_t(output.getNumSamples())});
    //     peak_normalization(std::span{output.getWritePointer(1), size_t(output.getNumSamples())});

    //     auto end = std::chrono::system_clock::now();
    //     std::cout << "TCONV(80): " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    //               << '\n';

    //     writeToWavFile(file, output, 44'100.0, 32);
    // }
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
    _openFile.setBounds(controls.removeFromLeft(controls.proportionOfWidth(0.5)));
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
    auto const load         = [this](juce::FileChooser const& chooser)
    {
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

}  // namespace neo
