#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/algorithm/fill.hpp>
#include <neo/algorithm/multiply.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/transform/rfft.hpp>
#include <neo/math/idiv.hpp>
#include <neo/math/windowing.hpp>

namespace neo::fft {

template<in_matrix InMat>
[[nodiscard]] auto stft(InMat buffer, int windowSize)
    -> stdex::mdarray<std::complex<typename InMat::value_type>, stdex::dextents<size_t, 2>>
{
    using Float = typename InMat::value_type;

    auto fft       = rfft_radix2_plan<Float>{ilog2(static_cast<size_t>(windowSize))};
    auto fftInput  = stdex::mdarray<Float, stdex::dextents<std::size_t, 1>>{fft.size()};
    auto fftOutput = stdex::mdarray<std::complex<Float>, stdex::dextents<std::size_t, 1>>{fft.size()};
    auto hann      = generate_window<Float>(static_cast<size_t>(windowSize));

    auto const totalNumSamples = static_cast<int>(buffer.extent(1));
    auto const numBins         = static_cast<std::size_t>(windowSize / 2 + 1);
    auto const numFrames       = static_cast<std::size_t>(idiv(totalNumSamples, windowSize));

    auto result = stdex::mdarray<std::complex<Float>, stdex::dextents<size_t, 2>>{numFrames, numBins};

    for (auto frameIdx{0UL}; frameIdx < result.extent(0); ++frameIdx) {
        auto const idx        = static_cast<int>(frameIdx * result.extent(1));
        auto const numSamples = std::min(totalNumSamples - idx, windowSize);
        auto const channel    = 0;

        auto block  = stdex::submdspan(buffer, channel, std::tuple{idx, idx + numSamples});
        auto window = stdex::submdspan(fftInput.to_mdspan(), std::tuple{0, numSamples});
        auto coeffs = stdex::submdspan(fftOutput.to_mdspan(), std::tuple{0, result.extent(1)});
        auto frame  = stdex::submdspan(result.to_mdspan(), frameIdx, std::tuple{0, result.extent(1)});

        fill(fftInput.to_mdspan(), Float(0));
        fill(fftOutput.to_mdspan(), Float(0));

        multiply(block, hann.to_mdspan(), window);
        fft(window, fftOutput.to_mdspan());

        scale(Float(1) / static_cast<Float>(windowSize), coeffs);
        copy(coeffs, frame);
    }

    return result;
}

}  // namespace neo::fft
