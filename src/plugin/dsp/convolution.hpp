#pragma once

#include "dsp/cola.hpp"
#include "dsp/resample.hpp"

#include "neo/fft.hpp"
#include "neo/fft/convolution.hpp"

#include <algorithm>
#include <juce_dsp/juce_dsp.h>
#include <vector>

namespace neo {

inline auto uniform_partition(juce::AudioBuffer<float> const& buffer, std::integral auto blockSize)
    -> stdex::mdarray<std::complex<float>, stdex::dextents<size_t, 3>>
{
    auto matrix = to_mdarray(buffer);
    return neo::fft::uniform_partition(matrix.to_mdspan(), static_cast<std::size_t>(blockSize));
}

struct Convolution
{
    using SampleType = float;

    Convolution() = default;

    auto loadImpulseResponse(std::unique_ptr<juce::InputStream> stream) -> void
    {
        jassert(stream != nullptr);

        auto formatManager = juce::AudioFormatManager{};
        formatManager.registerBasicFormats();

        auto reader = std::unique_ptr<juce::AudioFormatReader>(formatManager.createReaderFor(std::move(stream)));
        if (reader == nullptr) {
            return;
        }

        auto buffer = juce::AudioBuffer<float>{
            static_cast<int>(reader->numChannels),
            static_cast<int>(reader->lengthInSamples),
        };
        if (not reader->read(buffer.getArrayOfWritePointers(), buffer.getNumChannels(), 0, buffer.getNumSamples())) {
            return;
        }

        _impulse.emplace(std::move(buffer), reader->sampleRate);
        if (_spec.has_value()) {
            prepare(*_spec);
        }
    }

    auto prepare(juce::dsp::ProcessSpec const& spec) -> void
    {
        _spec = spec;

        if (_impulse.has_value()) {
            _convolvers.resize(spec.numChannels);
            auto const resampled  = resample(_impulse->buffer, _impulse->sampleRate, spec.sampleRate);
            auto const partitions = uniform_partition(resampled, neo::next_power_of_two(spec.maximumBlockSize));
        }
    }

    auto process(juce::dsp::AudioBlock<float> const& block) -> void
    {
        jassert(_spec.has_value());
        jassert(_spec->numChannels == block.getNumChannels());
    }

    auto reset() -> void {}

private:
    struct Impulse
    {
        juce::AudioBuffer<float> buffer;
        double sampleRate;
    };

    std::optional<Impulse> _impulse;
    std::optional<juce::dsp::ProcessSpec> _spec;
    std::vector<neo::fft::upols_convolver<float>> _convolvers;
};

[[nodiscard]] auto dense_convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter)
    -> juce::AudioBuffer<float>;

[[nodiscard]] auto
sparse_convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter, float thresholdDB)
    -> juce::AudioBuffer<float>;

}  // namespace neo
