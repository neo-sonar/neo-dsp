#pragma once

#include "dsp/cola.hpp"

#include "neo/fft/convolution.hpp"
#include "neo/testing/testing.hpp"

#include <algorithm>
#include <juce_dsp/juce_dsp.h>
#include <vector>

namespace neo {

struct DenseConvolution
{
    using SampleType = float;

    DenseConvolution() = default;

    auto loadImpulseResponse(std::unique_ptr<juce::InputStream> stream) -> void;

    auto prepare(juce::dsp::ProcessSpec const& spec) -> void;
    auto process(juce::dsp::AudioBlock<float> const& block) -> void;
    auto reset() -> void;

private:
    struct ImpulseResponse
    {
        juce::AudioBuffer<float> buffer;
        double sampleRate;
    };

    auto updateImpulseResponse() -> void;

    std::optional<ImpulseResponse> _impulse;
    std::optional<juce::dsp::ProcessSpec> _spec;
    std::vector<neo::fft::upols_convolver<float>> _convolvers;
    stdex::mdarray<std::complex<float>, stdex::dextents<std::size_t, 2>> _filter;
};

[[nodiscard]] auto dense_convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter)
    -> juce::AudioBuffer<float>;

[[nodiscard]] auto
sparse_convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter, float thresholdDB)
    -> juce::AudioBuffer<float>;

}  // namespace neo
