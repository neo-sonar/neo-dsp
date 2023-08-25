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

[[nodiscard]] constexpr auto num_sftf_frames(int signal_len, int frame_len, int overlap_len) noexcept
{
    return 1 + idiv((signal_len - frame_len + overlap_len), (frame_len - overlap_len));
}

}  // namespace detail

struct stft_options
{
    int frame_length;
    int overlap_length;
    int transform_size;
};

template<std::floating_point Float>
struct stft_plan
{
    explicit stft_plan(int transform_size)
        : _options{
            .frame_length   = transform_size,
            .overlap_length = transform_size / 2,
            .transform_size = transform_size,
        }
    {}

    explicit stft_plan(stft_options options) : _options{options} {}

    template<in_matrix InMat>
    [[nodiscard]] auto operator()(InMat x)
    {
        auto const scalar            = Float(1) / static_cast<Float>(_rfft.size());
        auto const total_num_samples = static_cast<int>(x.extent(1));
        auto const num_bins          = _rfft.size() / 2UL + 1UL;
        auto const num_frames        = static_cast<std::size_t>(
            detail::num_sftf_frames(total_num_samples, _options.frame_length, _options.overlap_length)
        );

        auto result = stdex::mdarray<std::complex<Float>, stdex::dextents<size_t, 2>>{
            num_frames,
            num_bins,
        };

        for (auto frame_idx{0UL}; frame_idx < result.extent(0); ++frame_idx) {
            fill(_fft_input.to_mdspan(), Float(0));
            fill(_fft_output.to_mdspan(), Float(0));

            auto const idx         = static_cast<int>(frame_idx) * _options.overlap_length;
            auto const num_samples = std::min(total_num_samples - idx, _options.frame_length);
            auto const channel     = 0;

            auto block  = stdex::submdspan(x, channel, std::tuple{idx, idx + num_samples});
            auto window = stdex::submdspan(_fft_input.to_mdspan(), std::tuple{0, num_samples});

            copy(block, window);
            multiply(_fft_input.to_mdspan(), _hann.to_mdspan(), _fft_input.to_mdspan());
            _rfft(_fft_input.to_mdspan(), _fft_output.to_mdspan());

            auto coeffs = stdex::submdspan(_fft_output.to_mdspan(), std::tuple{0, result.extent(1)});
            auto frame  = stdex::submdspan(result.to_mdspan(), frame_idx, std::tuple{0, result.extent(1)});

            scale(scalar, coeffs);
            copy(coeffs, frame);
        }

        return result;
    }

private:
    stft_options _options;
    rfft_radix2_plan<Float> _rfft{static_cast<size_t>(ilog2(_options.transform_size))};
    stdex::mdarray<Float, stdex::dextents<std::size_t, 1>> _fft_input{_rfft.size()};
    stdex::mdarray<std::complex<Float>, stdex::dextents<std::size_t, 1>> _fft_output{_rfft.size()};
    stdex::mdarray<Float, stdex::dextents<std::size_t, 1>> _hann{generate_window<Float>(_rfft.size())};
};

template<in_matrix InMat>
[[nodiscard]] auto stft(InMat x, stft_options options)
{
    auto plan = stft_plan<typename InMat::value_type>{options};
    return plan(x);
}

template<in_matrix InMat>
[[nodiscard]] auto stft(InMat x, int window_size)
    -> stdex::mdarray<std::complex<typename InMat::value_type>, stdex::dextents<size_t, 2>>
{
    auto plan = stft_plan<typename InMat::value_type>{window_size};
    return plan(x);
}

}  // namespace neo::fft
