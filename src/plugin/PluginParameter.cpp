#include "PluginParameter.hpp"

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

namespace mc {

auto createParameters() -> juce::AudioProcessorValueTreeState::ParameterLayout
{

    // auto gainRange = juce::NormalisableRange(0.0F, 4.0F, 0.01F);
    // gainRange.setSkewForCentre(1.0F);

    // auto gainAttributes = juce::AudioParameterFloatAttributes()
    //                           .withStringFromValueFunction(GainTextConverter{})
    //                           .withValueFromStringFunction(GainTextConverter{});

    // auto filterTypes = juce::StringArray{"Low Cut", "Peak", "High Cut"};

    return {
        // makeFloatParameter(ParamID::inGain, "Input Gain", gainRange, 1.0F,
        // gainAttributes),
        // makeBoolParameter(ParamID::filterEnable, "Filter Enable", false),
        // makeChoiceParameter(ParamID::filterType, "Filter Type", filterTypes, 1),
        // makeFloatParameter(ParamID::outGain, "Output Gain", gainRange, 1.0F,
        // gainAttributes),
    };
}

}  // namespace mc
