#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/algorithm/fill.hpp>
#include <neo/algorithm/multiply.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/rfft.hpp>
#include <neo/math/bit_ceil.hpp>
#include <neo/math/ilog2.hpp>

#include <cassert>
#include <utility>

namespace neo {

template<in_vector Signal, in_vector Patch>
    requires(std::same_as<typename Signal::value_type, typename Patch::value_type>)
auto fft_convolve(Signal signal, Patch patch)
{
    using Float = typename Signal::value_type;

    if (signal.extent(0) == 0 or patch.extent(0) == 0) {
        return stdex::mdarray<Float, stdex::dextents<size_t, 1>>{};
    }

    auto const size      = static_cast<std::size_t>(signal.extent(0) + patch.extent(0) - 1);
    auto const fft_size  = bit_ceil(size);
    auto const fft_order = ilog2(fft_size);
    auto const num_coeff = fft_size / 2 + 1;

    auto plan = fft::rfft_plan<Float>{fft_order};

    auto tmp_buf = stdex::mdarray<Float, stdex::dextents<size_t, 1>>{fft_size};
    auto tmp     = tmp_buf.to_mdspan();

    auto signal_spectrum = stdex::mdarray<std::complex<Float>, stdex::dextents<size_t, 1>>{num_coeff};
    auto patch_spectrum  = stdex::mdarray<std::complex<Float>, stdex::dextents<size_t, 1>>{num_coeff};

    // Signal R2C
    copy(signal, stdex::submdspan(tmp, std::tuple{0, signal.extent(0)}));
    fill(stdex::submdspan(tmp, std::tuple{signal.extent(0), tmp.extent(0)}), Float(0));
    rfft(plan, tmp, signal_spectrum.to_mdspan());

    // Patch R2C
    copy(patch, stdex::submdspan(tmp, std::tuple{0, patch.extent(0)}));
    fill(stdex::submdspan(tmp, std::tuple{patch.extent(0), tmp.extent(0)}), Float(0));
    rfft(plan, tmp, patch_spectrum.to_mdspan());

    // Convolve
    multiply(signal_spectrum.to_mdspan(), patch_spectrum.to_mdspan(), signal_spectrum.to_mdspan());

    // C2R
    fill(tmp, Float(0));
    irfft(plan, signal_spectrum.to_mdspan(), tmp);
    scale(Float(1) / Float(fft_size), tmp);

    // Copy to output
    auto output = stdex::mdarray<Float, stdex::dextents<size_t, 1>>{size};
    copy(stdex::submdspan(tmp, std::tuple{0, size}), output.to_mdspan());
    return output;
}

}  // namespace neo
