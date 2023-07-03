#pragma once

#include "neo/math.hpp"
#include "neo/mdspan.hpp"

#include <juce_dsp/juce_dsp.h>
#include <juce_graphics/juce_graphics.h>

namespace neo
{

inline auto stft(juce::AudioBuffer<float> const& buffer, int windowSize)
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

inline auto
normalized_power_spectrum_image(Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> frames,
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
                // img.setPixelAt(c, r, juce::Colour::fromHSV(level, 1.0F, level, 1.0F));
            }

            // img.setPixelAt(c, r, juce::Colours::black);
            // if (dB <= threshold + 30.0F) { img.setPixelAt(c, r, juce::Colours::green); }
            // if (dB <= threshold + 20.0F) { img.setPixelAt(c, r, juce::Colours::red); }
            // if (dB <= threshold + 10.0F) { img.setPixelAt(c, r, juce::Colours::blue); }
            // if (dB <= threshold) { img.setPixelAt(c, r, juce::Colours::white); }
        }
    }

    return img;
}

inline auto minmax_bin(Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> frames)
    -> std::pair<float, float>
{
    auto min = 99999.0F;
    auto max = 0.0F;

    for (auto f{0U}; f < frames.extent(0); ++f)
    {
        for (auto b{0U}; b < frames.extent(1); ++b)
        {
            auto const amplitude = std::abs(frames(f, b));

            min = std::min(min, amplitude);
            max = std::max(max, amplitude);
        }
    }

    return std::make_pair(min, max);
}

inline auto count_below_threshold(Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> frames,
                                  float threshold) -> int
{
    auto max = 0.0F;
    for (auto f{0U}; f < frames.extent(0); ++f)
    {
        for (auto b{0U}; b < frames.extent(1); ++b)
        {
            auto const gain = std::abs(frames(f, b));
            max             = std::max(max, gain * gain);
        }
    }
    auto scale = 1.0F / max;

    auto count = 0;
    for (auto f{0U}; f < frames.extent(0); ++f)
    {
        for (auto b{0U}; b < frames.extent(1); ++b)
        {
            auto const bin = std::abs(frames(f, b));
            if (juce::Decibels::gainToDecibels(bin * bin * scale) <= threshold) { ++count; }
        }
    }
    return count;
}

}  // namespace neo
