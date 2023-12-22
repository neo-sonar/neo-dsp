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

[[nodiscard]] auto
dense_convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter, int blockSize)
    -> juce::AudioBuffer<float>;

[[nodiscard]] auto sparse_convolve(
    juce::AudioBuffer<float> const& signal,
    juce::AudioBuffer<float> const& filter,
    int blockSize,
    double sampleRate,
    float thresholdDB,
    int lowBinsToKeep
) -> juce::AudioBuffer<float>;

}  // namespace neo
