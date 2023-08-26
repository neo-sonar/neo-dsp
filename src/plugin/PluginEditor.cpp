#include "PluginEditor.hpp"

namespace neo {

PluginEditor::PluginEditor(PluginProcessor& p)
    : AudioProcessorEditor(&p)
    , _parameterTab{p}
    , _benchmarkTab{p, _formats}
    , _tabs{juce::TabbedButtonBar::Orientation::TabsAtTop}
{
    _formats.registerBasicFormats();

    auto bgColor = getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId);
    _tabs.addTab("Parameter", bgColor, std::addressof(_parameterTab), false);
    _tabs.addTab("Benchmark", bgColor, std::addressof(_benchmarkTab), false);

    _openFile.onClick = [this] { openFile(); };

    processSpecChanged(p.getProcessSpec());
    p.addProcessSpecListener(this);

    addAndMakeVisible(_openFile);
    addAndMakeVisible(_tabs);

    addAndMakeVisible(_sampleRateLabel);
    addAndMakeVisible(_blockSizeLabel);
    addAndMakeVisible(_numChannelsLabel);

    setResizable(true, true);
    setSize(1280, 720);

    _tooltipWindow->setMillisecondsBeforeTipAppears(750);
}

PluginEditor::~PluginEditor() noexcept
{
    dynamic_cast<PluginProcessor&>(processor).removeProcessSpecListener(this);
    setLookAndFeel(nullptr);
}

auto PluginEditor::paint(juce::Graphics& g) -> void
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

auto PluginEditor::resized() -> void
{
    auto area       = getLocalBounds();
    auto labelArea  = area.removeFromBottom(area.proportionOfHeight(0.05));
    auto labelWidth = labelArea.proportionOfWidth(1.0 / 3.0);

    _sampleRateLabel.setBounds(labelArea.removeFromLeft(labelWidth));
    _blockSizeLabel.setBounds(labelArea.removeFromLeft(labelWidth));
    _numChannelsLabel.setBounds(labelArea.removeFromLeft(labelWidth));

    _openFile.setBounds(area.removeFromTop(area.proportionOfHeight(0.1)).reduced(0, 5));
    _tabs.setBounds(area);
}

auto PluginEditor::processSpecChanged(juce::dsp::ProcessSpec const& spec) -> void
{
    _sampleRateLabel.setText("Samplerate: " + juce::String{spec.sampleRate}, juce::sendNotification);
    _blockSizeLabel.setText("Block-size: " + juce::String{spec.maximumBlockSize}, juce::sendNotification);
    _numChannelsLabel.setText("Channels: " + juce::String{spec.numChannels}, juce::sendNotification);
}

auto PluginEditor::openFile() -> void
{
    auto const* msg         = "Please select an impulse response";
    auto const homeDir      = juce::File::getSpecialLocation(juce::File::userMusicDirectory);
    auto const chooserFlags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;
    auto const load         = [this](juce::FileChooser const& chooser) {
        if (chooser.getResults().isEmpty()) {
            return;
        }

        auto const file     = chooser.getResult();
        auto const filename = file.getFileNameWithoutExtension();
        _benchmarkTab.setImpulseResponseFile(file);
    };

    _fileChooser = std::make_unique<juce::FileChooser>(msg, homeDir, _formats.getWildcardForAllFormats());
    _fileChooser->launchAsync(chooserFlags, load);
}

}  // namespace neo
