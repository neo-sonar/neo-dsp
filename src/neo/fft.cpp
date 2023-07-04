#include "fft.hpp"

#include "neo/math.hpp"

namespace neo
{

auto stft(juce::AudioBuffer<float> const& buffer, int windowSize)
    -> KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>
{
    auto const order   = juce::roundToInt(std::log2(windowSize));
    auto const numBins = windowSize / 2 + 1;

    auto fft    = juce::dsp::FFT{order};
    auto window = juce::dsp::WindowingFunction<float>{
        static_cast<size_t>(windowSize),
        juce::dsp::WindowingFunction<float>::hann,
    };

    auto winBuffer = std::vector<float>(size_t(windowSize));
    auto input     = std::vector<std::complex<float>>(size_t(windowSize));
    auto output    = std::vector<std::complex<float>>(size_t(windowSize));

    auto result = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>{
        static_cast<size_t>(div_round(buffer.getNumSamples(), windowSize)),
        numBins,
    };

    for (auto f{0UL}; f < result.extent(0); ++f)
    {
        auto const idx        = static_cast<int>(f * result.extent(1));
        auto const numSamples = std::min(buffer.getNumSamples() - idx, windowSize);

        std::fill(winBuffer.begin(), winBuffer.end(), 0.0F);
        for (auto i{0}; i < numSamples; ++i) { winBuffer[size_t(i)] = buffer.getSample(0, idx + i); }
        window.multiplyWithWindowingTable(winBuffer.data(), winBuffer.size());
        std::copy(winBuffer.begin(), winBuffer.end(), input.begin());

        std::fill(output.begin(), output.end(), 0.0F);
        fft.perform(input.data(), output.data(), false);
        std::transform(output.begin(), output.end(), output.begin(), [=](auto v) { return v / float(windowSize); });
        for (auto b{0UL}; b < result.extent(1); ++b) { result(f, b) = output[b]; }
    }

    return result;
}

[[nodiscard]] auto powerSpectrumImage(Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> frames,
                                      float threshold) -> juce::Image
{
    auto const rows = static_cast<int>(frames.extent(0));
    auto const cols = static_cast<int>(frames.extent(1));

    auto max = 0.0F;
    for (auto f{0U}; f < frames.extent(0); ++f)
    {
        for (auto b{0U}; b < frames.extent(1); ++b)
        {
            auto const bin = std::abs(frames(f, b));
            max            = std::max(max, bin * bin);
        }
    }
    auto scale = 1.0F / max;

    auto img = juce::Image{juce::Image::PixelFormat::ARGB, cols, rows, true};
    for (auto f{0U}; f < frames.extent(0); ++f)
    {
        for (auto b{0U}; b < frames.extent(1); ++b)
        {
            img.setPixelAt(int(b), int(f), juce::Colours::black);

            auto const bin = std::abs(frames(f, b));
            auto const dB  = std::max(juce::Decibels::gainToDecibels(bin * bin * scale, -144.0F), -144.0F);
            if (dB < threshold)
            {
                auto level = juce::jmap(dB, -144.0F, threshold, 0.0F, 1.0F);
                img.setPixelAt(int(b), int(f), juce::Colours::white.darker(level));
            }
        }
    }

    return img;
}

[[nodiscard]] auto powerSpectrumImage(juce::AudioBuffer<float> const& buffer, float threshold) -> juce::Image
{
    if (buffer.getNumSamples() == 0) { return {}; }

    auto copy = buffer;
    neo::juce_normalization(copy);

    auto const frames = stft(copy, 1024);
    return powerSpectrumImage(frames, threshold);
}

}  // namespace neo
