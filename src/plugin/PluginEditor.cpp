#include "PluginEditor.hpp"

namespace neo {

PluginEditor::PluginEditor(PluginProcessor& p)
    : AudioProcessorEditor(&p)
    , _parameterTab{p}
    , _tabs{juce::TabbedButtonBar::Orientation::TabsAtTop}
{
    _formats.registerBasicFormats();

    auto bgColor = getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId);
    _tabs.addTab("Parameter", bgColor, std::addressof(_parameterTab), false);
    _tabs.addTab("Benchmark", bgColor, std::addressof(_benchmarkTab), false);

    _openFile.onClick = [this] { openFile(); };

    addAndMakeVisible(_openFile);
    addAndMakeVisible(_tabs);

    setResizable(true, true);
    setSize(1280, 720);

    _tooltipWindow->setMillisecondsBeforeTipAppears(750);
}

PluginEditor::~PluginEditor() noexcept { setLookAndFeel(nullptr); }

auto PluginEditor::paint(juce::Graphics& g) -> void
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

auto PluginEditor::resized() -> void
{
    auto area = getLocalBounds();
    _openFile.setBounds(area.removeFromTop(area.proportionOfHeight(0.1)).reduced(0, 5));
    _tabs.setBounds(area);
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
