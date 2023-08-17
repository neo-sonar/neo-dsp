#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/algorithm/fill.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/transform/rfft.hpp>
#include <neo/math/divide_round_up.hpp>

#include <concepts>

namespace neo::fft {

template<in_matrix InMat>
[[nodiscard]] auto uniform_partition(InMat buffer, std::size_t block_size)
{
    using Float = typename InMat::value_type;

    auto const window_size    = block_size * 2;
    auto const num_bins       = block_size + 1;
    auto const num_channels   = buffer.extent(0);
    auto const num_partitions = divide_round_up(buffer.extent(1), block_size);

    auto partitions = stdex::mdarray<std::complex<Float>, stdex::dextents<std::size_t, 3>>{
        num_channels,
        num_partitions,
        num_bins,
    };

    auto rfft = rfft_radix2_plan<Float>{ilog2(window_size)};
    auto in   = stdex::mdarray<Float, stdex::dextents<std::size_t, 1>>{rfft.size()};
    auto out  = stdex::mdarray<std::complex<Float>, stdex::dextents<std::size_t, 1>>{rfft.size()};

    auto const input        = in.to_mdspan();
    auto const output       = out.to_mdspan();
    auto const scale_factor = Float(1) / static_cast<Float>(rfft.size());

    for (auto channel{0UL}; channel < num_channels; ++channel) {
        for (auto partition_idx{0UL}; partition_idx < num_partitions; ++partition_idx) {
            auto const idx         = partition_idx * block_size;
            auto const num_samples = std::min(buffer.extent(1) - idx, block_size);
            auto const block       = stdex::submdspan(buffer, channel, std::tuple{idx, idx + num_samples});

            fill(input, Float(0));
            fill(output, std::complex<Float>(0));
            copy(block, stdex::submdspan(input, std::tuple{0, num_samples}));
            rfft(input, output);

            auto const coeffs    = stdex::submdspan(output, std::tuple{0, num_bins});
            auto const partition = stdex::submdspan(partitions.to_mdspan(), channel, partition_idx, stdex::full_extent);
            copy(coeffs, partition);
            scale(scale_factor, partition);
        }
    }

    return partitions;
}

}  // namespace neo::fft
