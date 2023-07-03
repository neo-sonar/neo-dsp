#pragma once

#include <juce_dsp/juce_dsp.h>
#include <juce_graphics/juce_graphics.h>

#include <span>

namespace neo
{

inline auto stft(juce::AudioBuffer<float> const& buffer, int windowSize)
    -> std::vector<std::vector<std::complex<float>>>
{
    auto order  = juce::roundToInt(std::log2(windowSize));
    auto fft    = juce::dsp::FFT{order};
    auto window = juce::dsp::WindowingFunction<float>{size_t(windowSize), juce::dsp::WindowingFunction<float>::hann};

    auto tmp    = std::vector<float>(size_t(windowSize));
    auto input  = std::vector<std::complex<float>>(size_t(windowSize));
    auto frames = std::vector<std::vector<std::complex<float>>>{};
    for (auto i{0}; i < buffer.getNumSamples(); i += windowSize)
    {
        auto const samples = std::min(buffer.getNumSamples() - i, windowSize);
        std::fill(tmp.begin(), tmp.end(), 0.0F);
        for (auto j{0}; j < samples; ++j) { tmp[size_t(j)] = buffer.getSample(0, i + j); }
        window.multiplyWithWindowingTable(tmp.data(), tmp.size());
        std::copy(tmp.begin(), tmp.end(), input.begin());

        auto out = std::vector<std::complex<float>>(size_t(windowSize));
        std::transform(out.begin(), out.end(), out.begin(), [=](auto v) { return v / float(windowSize); });
        fft.perform(input.data(), out.data(), false);
        out.resize(out.size() / 2U + 1U);
        frames.push_back(std::move(out));
    }

    return frames;
}

inline auto normalized_power_spectrum_image(std::span<std::vector<std::complex<float>> const> frames, float threshold)
    -> juce::Image
{
    auto const cols = static_cast<int>(frames[0].size());
    auto const rows = static_cast<int>(frames.size());

    auto max = 0.0F;
    for (auto r{0}; r < rows; ++r)
    {
        for (auto c{0}; c < cols; ++c)
        {
            auto const bin = std::abs(frames[size_t(r)][size_t(c)]);
            max            = std::max(max, bin * bin);
        }
    }
    auto scale = 1.0F / max;

    auto img = juce::Image{juce::Image::PixelFormat::ARGB, cols, rows, true};
    for (auto r{0}; r < rows; ++r)
    {
        for (auto c{0}; c < cols; ++c)
        {
            img.setPixelAt(c, r, juce::Colours::black);

            auto const bin = std::abs(frames[size_t(r)][size_t(c)]);
            auto const dB  = std::max(juce::Decibels::gainToDecibels(bin * bin * scale), -100.0F);
            if (dB < threshold)
            {
                auto level = juce::jmap(dB, -100.0F, threshold, 0.0F, 1.0F);
                img.setPixelAt(c, r, juce::Colours::white.darker(level));
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

inline auto minmax_bin(std::span<std::vector<std::complex<float>> const> frames) -> std::pair<float, float>
{
    auto min = 99999.0F;
    auto max = 0.0F;

    for (auto const& frame : frames)
    {
        for (auto const& bin : frame)
        {
            auto const amplitude = std::abs(bin);

            min = std::min(min, amplitude);
            max = std::max(max, amplitude);
        }
    }

    return std::make_pair(min, max);
}

inline auto count_below_threshold(std::span<std::vector<std::complex<float>> const> frames, float threshold) -> int
{
    auto max = 0.0F;
    for (auto const& frame : frames)
    {
        for (auto const& bin : frame)
        {
            auto const gain = std::abs(bin);
            max             = std::max(max, gain * gain);
        }
    }
    auto scale = 1.0F / max;

    auto count = 0;
    for (auto const& frame : frames)
    {
        for (auto const& bin : frame)
        {
            if (juce::Decibels::gainToDecibels(std::abs(bin) * std::abs(bin) * scale) <= threshold) { ++count; }
        }
    }
    return count;
}

}  // namespace neo
