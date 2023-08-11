#include "normalize.hpp"

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

}  // namespace neo
