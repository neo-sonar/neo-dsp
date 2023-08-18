#include "ParameterTab.hpp"

namespace neo {

ParameterTab::ParameterTab(juce::AudioProcessor& processor) : _editor{processor} { addAndMakeVisible(_editor); }

auto ParameterTab::paint(juce::Graphics& g) -> void { juce::ignoreUnused(g); }

auto ParameterTab::resized() -> void { _editor.setBounds(getLocalBounds()); }

}  // namespace neo
