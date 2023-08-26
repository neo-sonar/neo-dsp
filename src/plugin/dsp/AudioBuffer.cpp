
#include "AudioBuffer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace neo {

auto juce_normalization(juce::AudioBuffer<float>& buf) -> void
{
    auto calculateNormalisationFactor = [](float sumSquaredMagnitude) {
        if (sumSquaredMagnitude < 1e-8F)
            return 1.0F;
        return 0.125F / std::sqrt(sumSquaredMagnitude);
    };

    auto const numChannels = buf.getNumChannels();
    auto const numSamples  = buf.getNumSamples();
    auto const channelPtrs = buf.getArrayOfWritePointers();

    auto const maxSumSquaredMag
        = std::accumulate(channelPtrs, channelPtrs + numChannels, 0.0f, [numSamples](auto max, auto* channel) {
              auto const square_sum = [](auto sum, auto samp) { return sum + (samp * samp); };
              return std::max(max, std::accumulate(channel, channel + numSamples, 0.0f, square_sum));
          });

    auto const normalisationFactor = calculateNormalisationFactor(maxSumSquaredMag);

    std::for_each(channelPtrs, channelPtrs + numChannels, [normalisationFactor, numSamples](auto* channel) {
        juce::FloatVectorOperations::multiply(channel, normalisationFactor, numSamples);
    });
}

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
