#pragma once

#include <neo/container/mdspan.hpp>
#include <neo/fft/transform/stft.hpp>
#include <neo/math/windowing.hpp>

namespace neo::fft {

template<in_matrix InMat>
[[nodiscard]] auto uniform_partition(InMat impulse_response, std::size_t block_size)
{
    using Float = typename InMat::value_type;

    auto plan = stft_plan<Float>({
        .frame_length   = static_cast<int>(block_size),
        .transform_size = static_cast<int>(block_size * 2UL),
        .overlap_length = 0,
        .window         = rectangular_window<Float>{},
    });

    return plan(impulse_response);
}

}  // namespace neo::fft
