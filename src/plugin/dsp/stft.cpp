#include "stft.hpp"

#include "resample.hpp"

#include <neo/fft/math/divide_round_up.hpp>

#include <juce_audio_basics/juce_audio_basics.h>

#include <complex>

namespace neo::fft {

auto stft(Kokkos::mdspan<float const, Kokkos::dextents<size_t, 2>> buffer, int windowSize)
    -> KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>
{
    auto const order = juce::roundToInt(std::log2(windowSize));

    auto fft    = juce::dsp::FFT{order};
    auto window = juce::dsp::WindowingFunction<float>{
        static_cast<size_t>(windowSize),
        juce::dsp::WindowingFunction<float>::hann,
    };

    auto const totalNumSamples = static_cast<int>(buffer.extent(1));
    auto const numBins         = static_cast<std::size_t>(windowSize / 2 + 1);
    auto const numFrames       = static_cast<std::size_t>(divide_round_up(totalNumSamples, windowSize));

    auto winBuffer = std::vector<float>(size_t(windowSize));
    auto input     = std::vector<std::complex<float>>(size_t(windowSize));
    auto output    = std::vector<std::complex<float>>(size_t(windowSize));
    auto result    = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>{numFrames, numBins};

    for (auto f{0UL}; f < result.extent(0); ++f) {
        auto const idx        = static_cast<int>(f * result.extent(1));
        auto const numSamples = std::min(totalNumSamples - idx, windowSize);

        std::fill(winBuffer.begin(), winBuffer.end(), 0.0F);
        for (auto i{0}; i < numSamples; ++i) {
            winBuffer[size_t(i)] = buffer(0, idx + i);
        }
        window.multiplyWithWindowingTable(winBuffer.data(), winBuffer.size());
        std::copy(winBuffer.begin(), winBuffer.end(), input.begin());

        std::fill(output.begin(), output.end(), 0.0F);
        fft.perform(input.data(), output.data(), false);
        std::transform(output.begin(), output.end(), output.begin(), [=](auto v) { return v / float(windowSize); });
        for (auto b{0UL}; b < result.extent(1); ++b) {
            result(f, b) = output[b];
        }
    }

    return result;
}

}  // namespace neo::fft
