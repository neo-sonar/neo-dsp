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

auto powerSpectrumImage(Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> frames, float threshold)
    -> juce::Image
{
    auto const scale = [=]
    {
        auto max = 0.0F;
        for (auto f{0U}; f < frames.extent(0); ++f)
        {
            for (auto b{0U}; b < frames.extent(1); ++b)
            {
                auto const bin = std::abs(frames(f, b));
                max            = std::max(max, bin * bin);
            }
        }
        return 1.0F / max;
    }();

    auto const fillPixel = [=](auto& img, int x, int y, auto bin)
    {
        auto const power      = bin * bin;
        auto const normalized = power * scale;
        auto const dB         = juce::Decibels::gainToDecibels(normalized, -144.0F);
        auto const color      = [=]
        {
            if (dB < threshold)
            {
                auto level = juce::jmap(dB, -144.0F, threshold, 0.0F, 1.0F);
                return juce::Colours::white.darker(level);
            }
            return juce::Colours::black;
        }();

        img.setPixelAt(x, y, color);
    };

    auto const cols = static_cast<int>(frames.extent(1));
    auto const rows = static_cast<int>(frames.extent(0));
    auto img        = juce::Image{juce::Image::PixelFormat::ARGB, cols, rows, true};

    for (auto f{0U}; f < frames.extent(0); ++f)
    {
        for (auto b{0U}; b < frames.extent(1); ++b)
        {
            fillPixel(img, static_cast<int>(b), static_cast<int>(f), std::abs(frames(f, b)));
        }
    }

    return img;
}

auto powerHistogram(Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> frames) -> std::vector<int>
{
    auto const scale = [=]
    {
        auto max = 0.0F;
        for (auto f{0U}; f < frames.extent(0); ++f)
        {
            for (auto b{0U}; b < frames.extent(1); ++b)
            {
                auto const bin = std::abs(frames(f, b));
                max            = std::max(max, bin * bin);
            }
        }
        return 1.0F / max;
    }();

    auto histogram = std::vector<int>(144, 0);

    for (auto f{0U}; f < frames.extent(0); ++f)
    {
        for (auto b{0U}; b < frames.extent(1); ++b)
        {
            auto const bin        = std::abs(frames(f, b));
            auto const power      = bin * bin;
            auto const normalized = power * scale;
            auto const dB         = std::clamp(juce::Decibels::gainToDecibels(normalized, -144.0F), -143.0F, 0.0F);
            auto const index      = static_cast<std::size_t>(std::lround(std::abs(dB)));
            histogram[index] += 1;
        }
    }

    return histogram;
}

auto powerHistogramImage(Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> spectogram)
    -> juce::Image
{
    auto const histogram = powerHistogram(spectogram);
    auto const maxBin    = std::max_element(histogram.begin(), std::prev(histogram.end()));
    if (maxBin == histogram.end()) { return {}; }

    auto const binWidth  = 8;
    auto const imgWidth  = static_cast<int>((histogram.size() - 1) * binWidth);
    auto const imgHeight = 250;

    auto img = juce::Image{juce::Image::PixelFormat::ARGB, imgWidth, imgHeight, true};
    auto g   = juce::Graphics{img};
    g.fillAll(juce::Colours::black);

    for (auto i{0U}; i < histogram.size() - 1U; ++i)
    {
        auto const x      = int(i) * binWidth;
        auto const height = float(histogram[i]) / float(*maxBin) * float(imgHeight);
        auto const rect   = juce::Rectangle{x, 0, binWidth, juce::roundToInt(height)};

        g.setColour(juce::Colours::white);
        g.fillRect(rect.withBottomY(img.getBounds().getBottom()));

        if ((i == 32) || (i == 64) || (i == 96))
        {
            g.setColour(juce::Colours::red.withAlpha(0.65F));
            g.fillRect(rect.withHeight(img.getHeight()));
        }
    }

    return img;
}

auto powerSpectrumImage(juce::AudioBuffer<float> const& buffer, float threshold) -> juce::Image
{
    if (buffer.getNumSamples() == 0) { return {}; }

    auto copy = buffer;
    neo::juce_normalization(copy);

    auto const frames = stft(copy, 1024);
    return powerSpectrumImage(frames, threshold);
}

}  // namespace neo
