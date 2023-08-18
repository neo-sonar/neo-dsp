#include "PluginEditor.hpp"

namespace neo {

PluginEditor::PluginEditor(PluginProcessor& p)
    : AudioProcessorEditor(&p)
    , _parameter{p}
    , _tabs{juce::TabbedButtonBar::Orientation::TabsAtTop}
{
    auto bgColor = getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId);
    _tabs.addTab("Parameter", bgColor, std::addressof(_parameter), false);
    _tabs.addTab("Sparsity", bgColor, std::addressof(_sparsity), false);
    addAndMakeVisible(_tabs);

    setResizable(true, true);
    setSize(1024, 576);

    _tooltipWindow->setMillisecondsBeforeTipAppears(750);
}

PluginEditor::~PluginEditor() noexcept { setLookAndFeel(nullptr); }

auto PluginEditor::paint(juce::Graphics& g) -> void
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

auto PluginEditor::resized() -> void { _tabs.setBounds(getLocalBounds()); }

}  // namespace neo
