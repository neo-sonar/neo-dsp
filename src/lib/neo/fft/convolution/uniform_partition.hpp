#pragma once

#include "neo/fft/container/mdspan.hpp"
#include "neo/fft/math/divide_round_up.hpp"
#include "neo/fft/transform.hpp"

#include <complex>

namespace neo::fft {

inline auto uniform_partition(Kokkos::mdspan<float const, Kokkos::dextents<size_t, 2>> buffer, std::size_t blockSize)
    -> KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 3>>
{
    auto const windowSize    = blockSize * 2;
    auto const numBins       = windowSize / 2 + 1;
    auto const numChannels   = buffer.extent(0);
    auto const numPartitions = divide_round_up(buffer.extent(1), blockSize);

    auto partitions = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 3>>{
        numChannels,
        numPartitions,
        numBins,
    };

    auto input  = std::vector<float>(windowSize);
    auto output = std::vector<std::complex<float>>(windowSize);
    auto rfft   = rfft_radix2_plan<float>{ilog2(windowSize)};

    for (auto channel{0UL}; channel < numChannels; ++channel) {
        for (auto partition{0UL}; partition < numPartitions; ++partition) {
            auto const idx        = partition * blockSize;
            auto const numSamples = std::min(buffer.extent(1) - idx, blockSize);

            std::fill(input.begin(), input.end(), 0.0F);
            for (auto i{0UL}; i < numSamples; ++i) { input[i] = buffer(channel, idx + i); }

            std::fill(output.begin(), output.end(), 0.0F);
            rfft(input, output);
            for (auto bin{0UL}; bin < numBins; ++bin) {
                partitions(channel, partition, bin) = output[bin] / static_cast<float>(windowSize);
            }
        }
    }

    return partitions;
}

}  // namespace neo::fft
