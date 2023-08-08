#include "stft.hpp"

#include "dsp/normalize.hpp"
#include "neo/convolution/math/divide_round_up.hpp"

#include <complex>
#include <juce_audio_basics/juce_audio_basics.h>

namespace neo::fft {

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
        static_cast<size_t>(divide_round_up(buffer.getNumSamples(), windowSize)),
        numBins,
    };

    for (auto f{0UL}; f < result.extent(0); ++f) {
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
}  // namespace neo::fft
