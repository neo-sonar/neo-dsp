#pragma once

#include <juce_audio_formats/juce_audio_formats.h>

namespace neo
{

inline auto writeToWavFile(juce::File const& file, juce::AudioBuffer<float> const& buffer, double sampleRate,
                           int bitsPerSample) -> void
{

    auto const sr   = sampleRate;
    auto const bits = bitsPerSample;
    auto const ch   = static_cast<unsigned>(buffer.getNumChannels());

    auto wav    = juce::WavAudioFormat{};
    auto out    = file.createOutputStream();
    auto writer = std::unique_ptr<juce::AudioFormatWriter>{wav.createWriterFor(out.get(), sr, ch, bits, {}, 0)};

    if (writer != nullptr)
    {
        out.release();
        writer->writeFromAudioSampleBuffer(buffer, 0, buffer.getNumSamples());
    }
}

}  // namespace neo
