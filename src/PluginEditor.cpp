#include "PluginEditor.hpp"

#include "neo/convolution.hpp"
#include "neo/fft.hpp"
#include "neo/math.hpp"
#include "neo/resample.hpp"

#include <juce_dsp/juce_dsp.h>

#include <format>
#include <span>

namespace neo
{

PluginEditor::PluginEditor(PluginProcessor& p) : AudioProcessorEditor(&p)
{
    _formats.registerBasicFormats();

    _openFile.onClick = [this] { openFile(); };
    _threshold.setRange({-144.0, -10.0}, 0.0);
    _threshold.setValue(-90.0, juce::dontSendNotification);
    _threshold.onDragEnd = [this]
    {
        auto img = powerSpectrumImage(_spectrum, static_cast<float>(_threshold.getValue()));
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

    auto const signalFile = juce::File{R"(C:\Users\tobias\Music\Loops\Drums.wav)"};
    auto const filterFile = juce::File{R"(C:\Users\tobias\Music\Samples\IR\LexiconPCM90 Halls\LIVE_cannon gate.wav)"};

    auto const signal = loadAndResample(_formats, signalFile, 44'100.0);
    auto const filter = loadAndResample(_formats, filterFile, 44'100.0);
    auto convolved    = convolve(signal, filter);
    DBG(convolved.getNumSamples());

    peak_normalization(std::span{convolved.getWritePointer(0), size_t(convolved.getNumSamples())});

    auto wav = juce::WavAudioFormat{};
    auto out = juce::File{R"(C:\Users\tobias\Music)"}.getNonexistentChildFile("TestConv", ".wav").createOutputStream();
    auto writer = std::unique_ptr<juce::AudioFormatWriter>{
        wav.createWriterFor(out.get(), 44'100.0, static_cast<unsigned>(convolved.getNumChannels()), 16, {}, 0),
    };

    if (writer != nullptr)
    {
        out.release();
        writer->writeFromAudioSampleBuffer(convolved, 0, convolved.getNumSamples());
    }
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
        _spectrum = stft(_impulse, 1024);

        _spectogramImage.setImage(powerSpectrumImage(_spectrum, static_cast<float>(_threshold.getValue())));
        _histogramImage.setImage(powerHistogramImage(_spectrum));

        _fileInfo.setText(filename + " (" + juce::String(_impulse.getNumSamples()) + ")");

        repaint();
    };

    _fileChooser = std::make_unique<juce::FileChooser>(msg, homeDir, _formats.getWildcardForAllFormats());
    _fileChooser->launchAsync(chooserFlags, load);
}

}  // namespace neo
