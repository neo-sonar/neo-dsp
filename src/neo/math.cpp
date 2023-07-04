#include "math.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace neo
{

namespace
{

[[nodiscard]] auto peak_normalization_factor(std::span<float const> buf) -> float
{
    if (buf.empty()) { return 1.0F; }

    auto const abs_less = [](auto l, auto r) { return std::abs(l) < std::abs(r); };
    auto const max      = std::max_element(buf.begin(), buf.end(), abs_less);
    if (max == buf.end()) { return 1.0F; }

    auto const factor = 1.0F / std::abs(*max);
    return factor;
}

[[nodiscard]] auto rms_normalization_factor(std::span<float const> buf) -> float
{
    auto const squared_sum = [](auto sum, auto val) { return sum + (val * val); };
    auto const sum         = std::accumulate(buf.begin(), buf.end(), 0.0F, squared_sum);
    auto const mean_square = sum / static_cast<float>(buf.size());

    auto factor = 1.0F;
    if (mean_square > 0.0F) { factor = 1.0F / std::sqrt(mean_square); }
    return factor;
}

}  // namespace

// normalized_sample = sample / max(abs(buffer))
auto peak_normalization(std::span<float> buffer) -> void
{
    auto const factor   = peak_normalization_factor(buffer);
    auto const multiply = [factor](auto sample) { return sample * factor; };
    std::transform(buffer.begin(), buffer.end(), buffer.begin(), multiply);
}

// normalized_sample = sample / sqrt(mean(buffer^2))
auto rms_normalization(std::span<float> buffer) -> void
{
    if (buffer.empty()) return;
    auto const factor = rms_normalization_factor(buffer);
    auto const mul    = [factor](auto v) { return v * factor; };
    std::transform(buffer.begin(), buffer.end(), buffer.begin(), mul);
}

auto juce_normalization(juce::AudioBuffer<float>& buf) -> void
{
    auto calculateNormalisationFactor = [](float sumSquaredMagnitude)
    {
        if (sumSquaredMagnitude < 1e-8F) return 1.0F;
        return 0.125F / std::sqrt(sumSquaredMagnitude);
    };

    auto const numChannels = buf.getNumChannels();
    auto const numSamples  = buf.getNumSamples();
    auto const channelPtrs = buf.getArrayOfWritePointers();

    auto const maxSumSquaredMag
        = std::accumulate(channelPtrs, channelPtrs + numChannels, 0.0f,
                          [numSamples](auto max, auto* channel)
                          {
                              auto const square_sum = [](auto sum, auto samp) { return sum + (samp * samp); };
                              return std::max(max, std::accumulate(channel, channel + numSamples, 0.0f, square_sum));
                          });

    auto const normalisationFactor = calculateNormalisationFactor(maxSumSquaredMag);

    std::for_each(channelPtrs, channelPtrs + numChannels,
                  [normalisationFactor, numSamples](auto* channel)
                  { juce::FloatVectorOperations::multiply(channel, normalisationFactor, numSamples); });
}

}  // namespace neo
