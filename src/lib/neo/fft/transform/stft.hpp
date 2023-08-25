#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/algorithm/fill.hpp>
#include <neo/algorithm/multiply.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/transform/rfft.hpp>
#include <neo/math/idiv.hpp>
#include <neo/math/windowing.hpp>

#include <functional>

namespace neo::fft {

namespace detail {

[[nodiscard]] constexpr auto num_sftf_frames(int signal_len, int frame_len, int overlap_len) noexcept
{
    return 1 + idiv((signal_len - frame_len + overlap_len), (frame_len - overlap_len));
}

}  // namespace detail

template<std::floating_point Float>
struct stft_options
{
    int frame_length;
    int transform_size;
    int overlap_length;
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
        fill_window(_window_func.to_mdspan(), _options.window);
    }

    template<in_matrix InMat>
    [[nodiscard]] auto operator()(InMat x)
    {
        auto const total_num_samples = static_cast<int>(x.extent(1));
        auto const num_bins          = _rfft.size() / 2UL + 1UL;
        auto const num_frames        = static_cast<std::size_t>(
            detail::num_sftf_frames(total_num_samples, _options.frame_length, _options.overlap_length)
        );

        auto result = stdex::mdarray<std::complex<Float>, stdex::dextents<size_t, 3>>{
            x.extent(0),
            num_frames,
            num_bins,
        };

        for (auto channel{0}; std::cmp_less(channel, result.extent(0)); ++channel) {
            for (auto frame_idx{0}; std::cmp_less(frame_idx, result.extent(1)); ++frame_idx) {
                fill(_input.to_mdspan(), Float(0));
                fill(_output.to_mdspan(), Float(0));

                auto const sample_idx  = frame_idx * _options.frame_length - frame_idx * _options.overlap_length;
                auto const num_samples = std::min(total_num_samples - sample_idx, _options.frame_length);
                auto const block       = stdex::submdspan(x, channel, std::tuple{sample_idx, sample_idx + num_samples});
                auto const window      = stdex::submdspan(_input.to_mdspan(), std::tuple{0, num_samples});
                copy(block, window);

                multiply(_input.to_mdspan(), _window_func.to_mdspan(), _input.to_mdspan());
                _rfft(_input.to_mdspan(), _output.to_mdspan());

                auto coeffs = stdex::submdspan(_output.to_mdspan(), std::tuple{0, result.extent(2)});
                auto frame  = stdex::submdspan(result.to_mdspan(), channel, frame_idx, std::tuple{0, result.extent(2)});

                copy(coeffs, frame);
            }
        }

        return result;
    }

private:
    stft_options<Float> _options;

    rfft_radix2_plan<Float> _rfft{static_cast<size_t>(ilog2(_options.transform_size))};
    stdex::mdarray<Float, stdex::dextents<std::size_t, 1>> _input{_rfft.size()};
    stdex::mdarray<std::complex<Float>, stdex::dextents<std::size_t, 1>> _output{_rfft.size()};

    stdex::mdarray<Float, stdex::dextents<std::size_t, 1>> _window_func{_rfft.size()};
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
