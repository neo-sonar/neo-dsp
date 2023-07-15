#include "wav.hpp"

#include "dsp/resample.hpp"

namespace neo {

auto writeToWavFile(
    juce::File const& file,
    Kokkos::mdspan<float, Kokkos::dextents<size_t, 2>> buffer,
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
    for (auto i{0U}; i < ch; ++i) { channels[i] = std::addressof(buffer(i, 0)); }

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
