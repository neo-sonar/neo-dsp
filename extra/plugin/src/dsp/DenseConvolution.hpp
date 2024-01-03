// SPDX-License-Identifier: MIT

#pragma once

#include "dsp/AudioBuffer.hpp"
#include "dsp/ConstantOverlapAdd.hpp"

#include <neo/convolution.hpp>
#include <neo/testing/testing.hpp>

#include <algorithm>
#include <juce_dsp/juce_dsp.h>
#include <vector>

namespace neo {

struct DenseConvolution final : ConstantOverlapAdd<float>
{
    using SampleType = float;

    explicit DenseConvolution(int blockSize);
    ~DenseConvolution() override = default;

    auto loadImpulseResponse(std::unique_ptr<juce::InputStream> stream) -> void;

    auto prepareFrame(juce::dsp::ProcessSpec const& spec) -> void override;
    auto processFrame(juce::dsp::ProcessContextReplacing<float> const& context) -> void override;
    auto resetFrame() -> void override;

private:
    auto updateImpulseResponse() -> void;

    std::optional<BufferWithSampleRate<float>> _impulse;
    std::optional<juce::dsp::ProcessSpec> _spec;
    std::vector<neo::upols_convolver<std::complex<float>>> _convolvers;
    stdex::mdarray<std::complex<float>, stdex::dextents<std::size_t, 3>> _filter;
};

template<typename Convolver>
[[nodiscard]] auto
dense_convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter, int blockSize)
    -> juce::AudioBuffer<float>
{
    auto output = juce::AudioBuffer<float>{signal.getNumChannels(), signal.getNumSamples()};
    auto block  = std::vector<float>(size_t(blockSize));
    auto matrix = to_mdarray(filter);
    normalize_impulse(matrix.to_mdspan());
    auto partitions = uniform_partition(matrix.to_mdspan(), static_cast<std::size_t>(blockSize));

    for (auto ch{0}; ch < signal.getNumChannels(); ++ch) {
        auto convolver     = Convolver{};
        auto const channel = static_cast<size_t>(ch);
        auto const full    = stdex::full_extent;
        convolver.filter(stdex::submdspan(partitions.to_mdspan(), channel, full, full));

        auto const* const in = signal.getReadPointer(ch);
        auto* const out      = output.getWritePointer(ch);

        for (auto i{0}; i < output.getNumSamples(); i += blockSize) {
            auto const numSamples = std::min(output.getNumSamples() - i, blockSize);
            std::fill(block.begin(), block.end(), 0.0F);
            std::copy(std::next(in, i), std::next(in, i + numSamples), block.begin());
            convolver(stdex::mdspan{block.data(), stdex::extents{block.size()}});
            std::copy(block.begin(), std::next(block.begin(), numSamples), std::next(out, i));
        }
    }

    return output;
}

[[nodiscard]] auto sparse_convolve(
    juce::AudioBuffer<float> const& signal,
    juce::AudioBuffer<float> const& filter,
    int blockSize,
    double sampleRate,
    float thresholdDB,
    int lowBinsToKeep
) -> juce::AudioBuffer<float>;

}  // namespace neo
