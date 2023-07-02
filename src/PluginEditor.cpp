#include "PluginEditor.hpp"

PluginEditor::PluginEditor(PluginProcessor& p) : AudioProcessorEditor(&p)
{
    _tooltipWindow->setMillisecondsBeforeTipAppears(750);
    setSize(600, 400);
}

PluginEditor::~PluginEditor() noexcept { setLookAndFeel(nullptr); }

auto PluginEditor::paint(juce::Graphics& g) -> void
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

auto PluginEditor::resized() -> void {}
