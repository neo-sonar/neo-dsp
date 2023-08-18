#include "AudioFile.hpp"

#include "dsp/AudioBuffer.hpp"

namespace neo {

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

auto writeToWavFile(
    juce::File const& file,
    stdex::mdspan<float, stdex::dextents<size_t, 2>> buffer,
    double sampleRate,
    int bitsPerSample
) -> void
{

    auto const sr   = sampleRate;
    auto const bits = bitsPerSample;
    auto const ch   = static_cast<unsigned>(buffer.extent(0));

    auto wav    = juce::WavAudioFormat{};
    auto out    = file.createOutputStream();
    auto writer = std::unique_ptr<juce::AudioFormatWriter>{wav.createWriterFor(out.get(), sr, ch, bits, {}, 0)};

    auto channels = std::vector<float const*>(ch);
    for (auto i{0U}; i < ch; ++i) {
        channels[i] = std::addressof(buffer(i, 0));
    }

    if (writer != nullptr) {
        out.release();
        writer->writeFromFloatArrays(channels.data(), static_cast<int>(ch), static_cast<int>(buffer.extent(1)));
    }
}

auto writeToWavFile(
    juce::File const& file,
    juce::AudioBuffer<float> const& buffer,
    double sampleRate,
    int bitsPerSample
) -> void
{

    writeToWavFile(file, to_mdarray(buffer), sampleRate, bitsPerSample);
}

}  // namespace neo
