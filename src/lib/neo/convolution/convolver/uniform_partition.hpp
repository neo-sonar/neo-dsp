#pragma once

#include "neo/convolution/container/mdspan.hpp"
#include "neo/convolution/math/divide_round_up.hpp"
#include "neo/fft.hpp"

namespace neo::fft {

inline auto uniform_partition(Kokkos::mdspan<float const, Kokkos::dextents<size_t, 2>> buffer, std::size_t blockSize)
    -> KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 3>>
{
    auto const windowSize = blockSize * 2;
    auto const numBins    = windowSize / 2 + 1;

    auto partitions = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 3>>{
        buffer.extent(0),
        divide_round_up(buffer.extent(1), blockSize),
        numBins,
    };

    auto rfft   = rfft_radix2_plan<float>{ilog2(static_cast<size_t>(windowSize))};
    auto input  = std::vector<float>(size_t(windowSize));
    auto output = std::vector<std::complex<float>>(size_t(windowSize));

    for (auto channel{0UL}; channel < partitions.extent(0); ++channel) {

        for (auto partition{0UL}; partition < partitions.extent(1); ++partition) {
            auto const idx        = partition * partitions.extent(2);
            auto const numSamples = std::min(buffer.extent(1) - idx, blockSize);

            std::fill(input.begin(), input.end(), 0.0F);
            for (auto i{0UL}; i < numSamples; ++i) { input[i] = buffer(channel, idx + i); }

            std::fill(output.begin(), output.end(), 0.0F);
            rfft(input, output);
            for (auto bin{0UL}; bin < partitions.extent(2); ++bin) {
                partitions(channel, partition, bin) = output[bin] / static_cast<float>(windowSize);
            }
        }
    }

    return partitions;
}

}  // namespace neo::fft
