
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

auto resample(juce::AudioBuffer<float> const& buf, double srcSampleRate, double destSampleRate)
    -> juce::AudioBuffer<float>
{
    if (juce::exactlyEqual(srcSampleRate, destSampleRate)) {
        return buf;
    }

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
