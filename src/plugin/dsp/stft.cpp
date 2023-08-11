#include "stft.hpp"

#include "resample.hpp"

#include <neo/fft/algorithm/copy.hpp>
#include <neo/fft/algorithm/fill.hpp>
#include <neo/fft/algorithm/scale.hpp>
#include <neo/fft/math/divide_round_up.hpp>
#include <neo/fft/transform/rfft.hpp>

#include <juce_audio_basics/juce_audio_basics.h>

#include <complex>

namespace neo::fft {

auto stft(Kokkos::mdspan<float const, Kokkos::dextents<size_t, 2>> buffer, int windowSize)
    -> KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>
{
    auto fft            = rfft_plan<float>{ilog2(static_cast<size_t>(windowSize))};
    auto fftInput       = KokkosEx::mdarray<float, Kokkos::dextents<std::size_t, 1>>{fft.size()};
    auto fftOutput      = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<std::size_t, 1>>{fft.size()};
    auto windowFunction = juce::dsp::WindowingFunction<float>{
        static_cast<size_t>(windowSize),
        juce::dsp::WindowingFunction<float>::hann,
    };

    auto const totalNumSamples = static_cast<int>(buffer.extent(1));
    auto const numBins         = static_cast<std::size_t>(windowSize / 2 + 1);
    auto const numFrames       = static_cast<std::size_t>(divide_round_up(totalNumSamples, windowSize));

    auto result = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>{numFrames, numBins};

    for (auto f{0UL}; f < result.extent(0); ++f) {
        auto const idx        = static_cast<int>(f * result.extent(1));
        auto const numSamples = std::min(totalNumSamples - idx, windowSize);
        auto const channel    = 0;

        auto block  = KokkosEx::submdspan(buffer, channel, std::tuple{idx, idx + numSamples});
        auto window = KokkosEx::submdspan(fftInput.to_mdspan(), std::tuple{0, numSamples});
        auto coeffs = KokkosEx::submdspan(fftOutput.to_mdspan(), std::tuple{0, result.extent(1)});
        auto frame  = KokkosEx::submdspan(result.to_mdspan(), f, std::tuple{0, result.extent(1)});

        fill(fftInput.to_mdspan(), 0.0F);
        fill(fftOutput.to_mdspan(), 0.0F);
        copy(block, window);

        windowFunction.multiplyWithWindowingTable(window.data_handle(), window.extent(0));
        fft(window, fftOutput.to_mdspan());

        scale(1.0F / static_cast<float>(windowSize), coeffs);
        copy(coeffs, frame);
    }

    return result;
}

}  // namespace neo::fft
