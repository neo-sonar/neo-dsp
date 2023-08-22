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

template<complex Complex>
struct uniform_partitioner
{
    using value_type = Complex;
    using real_type  = typename Complex::value_type;
    using size_type  = std::size_t;

    explicit uniform_partitioner(size_type block_size) : _block_size{block_size} {}

    template<in_matrix InMat>
        requires std::convertible_to<typename InMat::value_type, real_type>
    auto operator()(InMat impulse_response)
    {
        auto const num_channels   = impulse_response.extent(0);
        auto const num_partitions = divide_round_up(impulse_response.extent(1), _block_size);

        auto partitions = stdex::mdarray<Complex, stdex::dextents<size_type, 3>>{
            num_channels,
            num_partitions,
            _num_bins,
        };

        auto const input  = _in.to_mdspan();
        auto const output = _out.to_mdspan();

        for (auto ch{0UL}; ch < num_channels; ++ch) {
            for (auto partition_idx{0UL}; partition_idx < num_partitions; ++partition_idx) {
                auto const idx         = partition_idx * _block_size;
                auto const num_samples = std::min(impulse_response.extent(1) - idx, _block_size);
                auto const block       = stdex::submdspan(impulse_response, ch, std::tuple{idx, idx + num_samples});

                fill(input, real_type(0));
                fill(output, Complex(0));
                copy(block, stdex::submdspan(input, std::tuple{0, num_samples}));
                _rfft(input, output);

                auto const coeffs    = stdex::submdspan(output, std::tuple{0, _num_bins});
                auto const partition = stdex::submdspan(partitions.to_mdspan(), ch, partition_idx, stdex::full_extent);
                copy(coeffs, partition);
            }
        }

        return partitions;
    }

private:
    size_type _block_size;
    size_type _window_size{_block_size * 2};
    size_type _num_bins{_block_size + 1};

    rfft_radix2_plan<real_type> _rfft{ilog2(_window_size)};
    stdex::mdarray<real_type, stdex::dextents<size_type, 1>> _in{_rfft.size()};
    stdex::mdarray<Complex, stdex::dextents<size_type, 1>> _out{_rfft.size()};
};

template<in_matrix InMat>
[[nodiscard]] auto uniform_partition(InMat impulse_response, std::size_t block_size)
{
    using Complex = std::complex<typename InMat::value_type>;
    return uniform_partitioner<Complex>{block_size}(impulse_response);
}

}  // namespace neo::fft
