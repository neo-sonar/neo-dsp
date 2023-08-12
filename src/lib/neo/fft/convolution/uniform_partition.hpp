#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/algorithm/fill.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/transform/rfft.hpp>
#include <neo/math/complex.hpp>
#include <neo/math/divide_round_up.hpp>

#include <concepts>

namespace neo::fft {

template<in_matrix InMat>
[[nodiscard]] auto uniform_partition(InMat buffer, std::size_t blockSize)
{
    using Float = typename InMat::value_type;

    auto const windowSize    = blockSize * 2;
    auto const numBins       = windowSize / 2 + 1;
    auto const numChannels   = buffer.extent(0);
    auto const numPartitions = divide_round_up(buffer.extent(1), blockSize);
    auto const scaleFactor   = Float(1) / static_cast<Float>(windowSize);

    auto partitions = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<std::size_t, 3>>{
        numChannels,
        numPartitions,
        numBins,
    };

    auto in   = KokkosEx::mdarray<Float, Kokkos::dextents<std::size_t, 1>>{windowSize};
    auto out  = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<std::size_t, 1>>{windowSize};
    auto rfft = rfft_radix2_plan<Float>{ilog2(windowSize)};

    auto const input  = in.to_mdspan();
    auto const output = out.to_mdspan();

    for (auto ch{0UL}; ch < numChannels; ++ch) {
        for (auto p{0UL}; p < numPartitions; ++p) {
            fill(input, Float(0));
            fill(output, std::complex<Float>(0));

            auto const idx        = p * blockSize;
            auto const numSamples = std::min(buffer.extent(1) - idx, blockSize);
            auto const block      = KokkosEx::submdspan(buffer, ch, std::tuple{idx, idx + numSamples});
            copy(block, KokkosEx::submdspan(input, std::tuple{0, numSamples}));

            rfft(input, output);

            auto const coeffs    = KokkosEx::submdspan(output, std::tuple{0, numBins});
            auto const partition = KokkosEx::submdspan(partitions.to_mdspan(), ch, p, Kokkos::full_extent);
            copy(coeffs, partition);
            scale(scaleFactor, partition);
        }
    }

    return partitions;
}

}  // namespace neo::fft
