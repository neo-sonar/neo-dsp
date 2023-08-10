
#include "resample.hpp"

namespace neo {

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

auto loadAndResample(juce::AudioFormatManager& formats, juce::File const& file, double sampleRate)
    -> juce::AudioBuffer<float>
{
    auto reader = std::unique_ptr<juce::AudioFormatReader>{formats.createReaderFor(file.createInputStream())};
    if (reader == nullptr) {
        return {};
    }

    auto buffer = juce::AudioBuffer<float>{int(reader->numChannels), int(reader->lengthInSamples)};
    if (!reader->read(buffer.getArrayOfWritePointers(), buffer.getNumChannels(), 0, buffer.getNumSamples())) {
        return {};
    }

    if (not juce::exactlyEqual(reader->sampleRate, sampleRate)) {
        return resample(buffer, reader->sampleRate, sampleRate);
    }
    return buffer;
}

auto to_mdarray(juce::AudioBuffer<float> const& buffer) -> KokkosEx::mdarray<float, Kokkos::dextents<std::size_t, 2>>
{
    auto result = KokkosEx::mdarray<float, Kokkos::dextents<std::size_t, 2>>{
        static_cast<std::size_t>(buffer.getNumChannels()),
        static_cast<std::size_t>(buffer.getNumSamples()),
    };

    for (auto ch{0ULL}; ch < result.extent(0); ++ch) {
        for (auto i{0ULL}; i < result.extent(1); ++i) {
            result(ch, i) = buffer.getSample(static_cast<int>(ch), static_cast<int>(i));
        }
    }

    return result;
}

}  // namespace neo
