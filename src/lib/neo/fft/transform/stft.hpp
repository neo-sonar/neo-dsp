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

namespace detail {

[[nodiscard]] constexpr auto num_sftf_segments(int signal_len, int segment_len, int overlap_len) noexcept
{
    return 1 + idiv((signal_len - segment_len + overlap_len), (segment_len - overlap_len));
}

static_assert(num_sftf_segments(1024, 128, 0) == 8);
static_assert(num_sftf_segments(1024, 256, 0) == 4);
static_assert(num_sftf_segments(1024, 256, 128) == 8);

}  // namespace detail

struct stft_options
{
    int segment_length;
    int overlap_length;
    int transform_size;
};

template<in_matrix InMat>
[[nodiscard]] auto stft(InMat x, stft_options options)
{
    assert(options.segment_length == options.transform_size);

    using Float = typename InMat::value_type;

    auto rfft       = rfft_radix2_plan<Float>{static_cast<size_t>(ilog2(options.transform_size))};
    auto fft_input  = stdex::mdarray<Float, stdex::dextents<std::size_t, 1>>{rfft.size()};
    auto fft_output = stdex::mdarray<std::complex<Float>, stdex::dextents<std::size_t, 1>>{rfft.size()};
    auto hann       = generate_window<Float>(rfft.size());
    auto scalar     = Float(1) / static_cast<Float>(rfft.size());

    auto const total_num_samples = static_cast<int>(x.extent(1));
    auto const num_bins          = rfft.size() / 2UL + 1UL;
    auto const num_segments      = static_cast<std::size_t>(
        detail::num_sftf_segments(total_num_samples, options.segment_length, options.overlap_length)
    );

    auto result = stdex::mdarray<std::complex<Float>, stdex::dextents<size_t, 2>>{num_segments, num_bins};

    for (auto segment_idx{0UL}; segment_idx < result.extent(0); ++segment_idx) {
        fill(fft_input.to_mdspan(), Float(0));
        fill(fft_output.to_mdspan(), Float(0));

        auto const idx         = static_cast<int>(segment_idx) * options.overlap_length;
        auto const num_samples = std::min(total_num_samples - idx, options.segment_length);
        auto const channel     = 0;

        auto block  = stdex::submdspan(x, channel, std::tuple{idx, idx + num_samples});
        auto window = stdex::submdspan(fft_input.to_mdspan(), std::tuple{0, num_samples});

        copy(block, window);
        multiply(fft_input.to_mdspan(), hann.to_mdspan(), fft_input.to_mdspan());
        rfft(fft_input.to_mdspan(), fft_output.to_mdspan());

        auto coeffs  = stdex::submdspan(fft_output.to_mdspan(), std::tuple{0, result.extent(1)});
        auto segment = stdex::submdspan(result.to_mdspan(), segment_idx, std::tuple{0, result.extent(1)});

        scale(scalar, coeffs);
        copy(coeffs, segment);
    }

    return result;
}

template<in_matrix InMat>
[[nodiscard]] auto stft(InMat x, int window_size)
    -> stdex::mdarray<std::complex<typename InMat::value_type>, stdex::dextents<size_t, 2>>
{
    return stft(
        x,
        stft_options{
            .segment_length = window_size,
            .overlap_length = window_size / 2,
            .transform_size = window_size,
        }
    );
}

}  // namespace neo::fft
