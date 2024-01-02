// SPDX-License-Identifier: MIT

#include "AudioBuffer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace neo {

auto resample(BufferWithSampleRate<float> const& buf, double destSampleRate) -> BufferWithSampleRate<float>
{
    if (juce::exactlyEqual(buf.sampleRate, destSampleRate)) {
        return buf;
    }

    auto const factorReading = buf.sampleRate / destSampleRate;

    auto original     = buf;
    auto memorySource = juce::MemoryAudioSource(original.buffer, false);

    auto const finalSize = juce::roundToInt(juce::jmax(1.0, buf.buffer.getNumSamples() / factorReading));

    auto result = BufferWithSampleRate<float>{
        .buffer     = juce::AudioBuffer<float>{buf.buffer.getNumChannels(), finalSize},
        .sampleRate = destSampleRate,
    };

    auto resampler = juce::ResamplingAudioSource(&memorySource, false, buf.buffer.getNumChannels());
    resampler.setResamplingRatio(factorReading);
    resampler.prepareToPlay(finalSize, buf.sampleRate);
    resampler.getNextAudioBlock({&result.buffer, 0, result.buffer.getNumSamples()});

    return result;
}

}  // namespace neo
