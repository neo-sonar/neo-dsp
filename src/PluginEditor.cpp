#include "PluginEditor.hpp"

#include "neo/fft.hpp"
#include "neo/math.hpp"
#include "neo/resample.hpp"

#include <juce_dsp/juce_dsp.h>

#include <format>
#include <span>

namespace neo
{

static auto summary(juce::AudioFormatReader& reader) -> std::pair<std::string, juce::Image>
{
    auto original = juce::AudioBuffer<float>{int(reader.numChannels), int(reader.lengthInSamples)};
    if (!reader.read(original.getArrayOfWritePointers(), original.getNumChannels(), 0, original.getNumSamples()))
    {
        throw std::runtime_error{"error reading"};
    }

    auto buffer       = resampleImpulseResponse(original, reader.sampleRate, 44100.0);
    auto const block  = juce::dsp::AudioBlock<float>{buffer};
    auto const minmax = block.findMinAndMax();

    neo::juce_normalization(buffer);
    // neo::peak_normalization(std::span<float>{buffer.getWritePointer(0), size_t(buffer.getNumSamples())});

    auto const minmax_n = block.findMinAndMax();
    auto const frames   = stft(buffer, 1024);

    auto const minmax_coeff   = minmax_bin(frames);
    auto const min_coeff_db   = juce::Decibels::gainToDecibels(minmax_coeff.first);
    auto const max_coeff_db   = juce::Decibels::gainToDecibels(minmax_coeff.second);
    auto const coeff_range_db = std::abs(min_coeff_db) - std::abs(max_coeff_db);

    auto const num_coeff = frames.size() * frames[0].size();
    auto const below_40  = count_below_threshold(frames, -40.0F);
    auto const below_50  = count_below_threshold(frames, -50.0F);
    auto const below_60  = count_below_threshold(frames, -60.0F);
    auto const below_70  = count_below_threshold(frames, -70.0F);
    auto const below_80  = count_below_threshold(frames, -80.0F);
    auto const below_90  = count_below_threshold(frames, -90.0F);

    auto const str = std::format(
        "length: {}\nch: {}\nsr: {}\nbits: {}\npeak: {}\nnorm_peak: {}\nframes: {}\ncoeff_min: {}\ncoeff_max: "
        "{}\nrange: {}\ncoeffs: {}\nbelow(-40dB): {} ({:.2f}%)\nbelow(-50dB): {} ({:.2f}%)\nbelow(-60dB): {} "
        "({:.2f}%)\nbelow(-70dB): {} ({:.2f}%)\nbelow(-80dB): {} ({:.2f}%)\nbelow(-90dB): {} ({:.2f}%)",
        reader.lengthInSamples, reader.numChannels, reader.sampleRate, reader.bitsPerSample,
        juce::Decibels::gainToDecibels(std::max(std::abs(minmax.getStart()), std::abs(minmax.getEnd()))),
        juce::Decibels::gainToDecibels(std::max(std::abs(minmax_n.getStart()), std::abs(minmax_n.getEnd()))),
        frames.size(), min_coeff_db, max_coeff_db, coeff_range_db, num_coeff, below_40,
        double(below_40) / double(num_coeff) * 100.0, below_50, double(below_50) / double(num_coeff) * 100.0, below_60,
        double(below_60) / double(num_coeff) * 100.0, below_70, double(below_70) / double(num_coeff) * 100.0, below_80,
        double(below_80) / double(num_coeff) * 100.0, below_90, double(below_90) / double(num_coeff) * 100.0);

    return std::make_pair(str, normalized_power_spectrum_image(frames, -90.0F));
}

PluginEditor::PluginEditor(PluginProcessor& p) : AudioProcessorEditor(&p)
{
    _formats.registerBasicFormats();

    _openFile.onClick = [this] { openFile(); };
    _fileInfo.setReadOnly(true);
    _fileInfo.setMultiLine(true);
    _image.setImagePlacement(juce::RectanglePlacement::centred);
    _tooltipWindow->setMillisecondsBeforeTipAppears(750);

    addAndMakeVisible(_openFile);
    addAndMakeVisible(_fileInfo);
    addAndMakeVisible(_image);

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
    _openFile.setBounds(bounds.removeFromTop(bounds.proportionOfHeight(0.1)));
    _fileInfo.setBounds(bounds.removeFromLeft(bounds.proportionOfWidth(0.25)));
    _image.setBounds(bounds);
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

        auto reader = std::unique_ptr<juce::AudioFormatReader>{_formats.createReaderFor(file.createInputStream())};
        if (reader == nullptr) { return; }

        auto s = summary(*reader);
        _fileInfo.setText(std::format("{}:\n{}\n\n", filename.toStdString(), s.first));
        _image.setImage(s.second);
        repaint();
    };

    _fileChooser = std::make_unique<juce::FileChooser>(msg, homeDir, _formats.getWildcardForAllFormats());
    _fileChooser->launchAsync(chooserFlags, load);
}

}  // namespace neo
