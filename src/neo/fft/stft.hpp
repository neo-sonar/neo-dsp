// SPDX-License-Identifier: MIT

#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/algorithm/fill.hpp>
#include <neo/algorithm/multiply.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/rfft.hpp>
#include <neo/math/idiv.hpp>
#include <neo/math/windowing.hpp>

#include <functional>

namespace neo::fft {

namespace detail {

[[nodiscard]] constexpr auto num_sftf_frames(int signal_len, int frame_len, int overlap_len) noexcept
{
    return 1 + idiv(signal_len - frame_len + overlap_len, frame_len - overlap_len);
}

}  // namespace detail

template<std::floating_point Float>
struct stft_options
{
    int frame_length{};
    int transform_size{};
    int overlap_length{};
    std::function<Float(int, int)> window{hann_window<Float>{}};
};

template<std::floating_point Float>
struct stft_plan
{
    explicit stft_plan(int transform_size)
        : stft_plan({
            .frame_length   = transform_size,
            .transform_size = transform_size,
            .overlap_length = transform_size / 2,
        })
    {}

    explicit stft_plan(stft_options<Float> options) : _options{std::move(options)}
    {
        fill_window(_window.to_mdspan(), _options.window);
    }

    template<in_matrix InMat>
    [[nodiscard]] auto operator()(InMat x)
    {
        auto const frame_len = _options.frame_length;
        auto const overlap   = _options.overlap_length;

        auto const num_channels = x.extent(0);
        auto const signal_len   = static_cast<int>(x.extent(1));

        auto const num_bins   = _rfft.size() / 2UL + 1UL;
        auto const num_frames = static_cast<std::size_t>(detail::num_sftf_frames(signal_len, frame_len, overlap));

        auto const in  = _input.to_mdspan();
        auto const out = _output.to_mdspan();

        auto result = stdex::mdarray<std::complex<Float>, stdex::dextents<size_t, 3>>{
            num_channels,
            num_frames,
            num_bins,
        };

        for (auto ch_idx{0}; std::cmp_less(ch_idx, num_channels); ++ch_idx) {
            for (auto frame_idx{0}; std::cmp_less(frame_idx, num_frames); ++frame_idx) {
                auto const sample_idx  = frame_idx * frame_len - frame_idx * overlap;
                auto const num_samples = std::min(signal_len - sample_idx, frame_len);

                auto const in_block  = stdex::submdspan(x, ch_idx, std::tuple{sample_idx, sample_idx + num_samples});
                auto const in_window = stdex::submdspan(in, std::tuple{0, num_samples});
                fill(in, Float(0));
                fill(out, Float(0));
                copy(in_block, in_window);

                multiply(in, _window.to_mdspan(), in);
                _rfft(in, out);

                auto coeffs       = stdex::submdspan(out, std::tuple{0, num_bins});
                auto result_frame = stdex::submdspan(result.to_mdspan(), ch_idx, frame_idx, std::tuple{0, num_bins});
                copy(coeffs, result_frame);
            }
        }

        return result;
    }

private:
    stft_options<Float> _options;

    rfft_plan<Float> _rfft{static_cast<size_t>(ilog2(_options.transform_size))};
    stdex::mdarray<Float, stdex::dextents<std::size_t, 1>> _input{_rfft.size()};
    stdex::mdarray<std::complex<Float>, stdex::dextents<std::size_t, 1>> _output{_rfft.size()};

    stdex::mdarray<Float, stdex::dextents<std::size_t, 1>> _window{_rfft.size()};
};

template<in_matrix InMat>
[[nodiscard]] auto stft(InMat x, stft_options<typename InMat::value_type> options)
{
    auto plan = stft_plan<typename InMat::value_type>{options};
    return plan(x);
}

template<in_matrix InMat>
[[nodiscard]] auto stft(InMat x, int window_size)
{
    auto plan = stft_plan<typename InMat::value_type>{window_size};
    return plan(x);
}

}  // namespace neo::fft
