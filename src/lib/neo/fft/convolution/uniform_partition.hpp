#pragma once

#include <neo/fft/container/mdspan.hpp>
#include <neo/fft/math/divide_round_up.hpp>
#include <neo/fft/transform/radix2.hpp>

#include <complex>
#include <concepts>

namespace neo::fft {

template<std::floating_point Float>
[[nodiscard]] auto
uniform_partition(Kokkos::mdspan<Float const, Kokkos::dextents<std::size_t, 2>> buffer, std::size_t blockSize)
    -> KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<std::size_t, 3>>
{
    auto const windowSize    = blockSize * 2;
    auto const numBins       = windowSize / 2 + 1;
    auto const numChannels   = buffer.extent(0);
    auto const numPartitions = divide_round_up(buffer.extent(1), blockSize);
    auto const scale         = Float(1) / static_cast<Float>(windowSize);

    auto partitions = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<std::size_t, 3>>{
        numChannels,
        numPartitions,
        numBins,
    };

    auto input  = std::vector<Float>(windowSize);
    auto output = std::vector<std::complex<Float>>(windowSize);
    auto rfft   = rfft_radix2_plan<Float>{ilog2(windowSize)};

    for (auto channel{0UL}; channel < numChannels; ++channel) {
        for (auto partition{0UL}; partition < numPartitions; ++partition) {
            auto const idx        = partition * blockSize;
            auto const numSamples = std::min(buffer.extent(1) - idx, blockSize);

            std::fill(input.begin(), input.end(), Float(0));
            for (auto i{0UL}; i < numSamples; ++i) { input[i] = buffer(channel, idx + i); }

            std::fill(output.begin(), output.end(), Float(0));
            rfft(input, output);
            for (auto bin{0UL}; bin < numBins; ++bin) { partitions(channel, partition, bin) = output[bin] * scale; }
        }
    }

    return partitions;
}

}  // namespace neo::fft
