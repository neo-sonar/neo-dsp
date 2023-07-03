#pragma once

#include <juce_audio_formats/juce_audio_formats.h>

namespace neo
{

inline auto resampleImpulseResponse(juce::AudioBuffer<float> const& buf, double const srcSampleRate,
                                    double const destSampleRate) -> juce::AudioBuffer<float>
{
    if (srcSampleRate == destSampleRate) return buf;

    auto const factorReading = srcSampleRate / destSampleRate;

    juce::AudioBuffer<float> original = buf;
    juce::MemoryAudioSource memorySource(original, false);
    juce::ResamplingAudioSource resamplingSource(&memorySource, false, buf.getNumChannels());

    auto const finalSize = juce::roundToInt(juce::jmax(1.0, buf.getNumSamples() / factorReading));
    resamplingSource.setResamplingRatio(factorReading);
    resamplingSource.prepareToPlay(finalSize, srcSampleRate);

    juce::AudioBuffer<float> result(buf.getNumChannels(), finalSize);
    resamplingSource.getNextAudioBlock({&result, 0, result.getNumSamples()});

    return result;
}

}  // namespace neo
