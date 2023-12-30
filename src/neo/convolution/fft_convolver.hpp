// SPDX-License-Identifier: MIT
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

template<std::floating_point Float>
struct fft_convolver
{
    fft_convolver(std::size_t signal_size, std::size_t patch_size) : _signal_size{signal_size}, _patch_size{patch_size}
    {
        assert(_signal_size > 1);
        assert(_patch_size > 1);
    }

    [[nodiscard]] auto signal_size() const noexcept -> std::size_t { return _signal_size; }

    [[nodiscard]] auto patch_size() const noexcept -> std::size_t { return _patch_size; }

    [[nodiscard]] auto output_size() const noexcept -> std::size_t { return _output_size; }

    template<in_vector Signal, in_vector Patch, out_vector Output>
    auto operator()(Signal signal, Patch patch, Output output) -> void
    {
        assert(signal.extent(0) == signal_size());
        assert(patch.extent(0) == patch_size());
        assert(output.extent(0) == output_size());

        auto const tmp             = _tmp.to_mdspan();
        auto const signal_spectrum = _signal_spectrum.to_mdspan();
        auto const patch_spectrum  = _patch_spectrum.to_mdspan();

        // Signal R2C
        copy(signal, stdex::submdspan(tmp, std::tuple{0, signal.extent(0)}));
        fill(stdex::submdspan(tmp, std::tuple{signal.extent(0), tmp.extent(0)}), Float(0));
        rfft(_plan, tmp, signal_spectrum);

        // Patch R2C
        copy(patch, stdex::submdspan(tmp, std::tuple{0, patch.extent(0)}));
        fill(stdex::submdspan(tmp, std::tuple{patch.extent(0), tmp.extent(0)}), Float(0));
        rfft(_plan, tmp, patch_spectrum);

        // Convolve
        multiply(signal_spectrum, patch_spectrum, signal_spectrum);

        // C2R
        fill(tmp, Float(0));
        irfft(_plan, signal_spectrum, tmp);
        scale(Float(1) / Float(_plan.size()), tmp);

        // Copy to output
        copy(stdex::submdspan(tmp, std::tuple{0, output_size()}), output);
    }

private:
    std::size_t _signal_size;
    std::size_t _patch_size;
    std::size_t _output_size{signal_size() + patch_size() - 1};
    fft::rfft_plan<Float> _plan{ilog2(bit_ceil(output_size()))};

    stdex::mdarray<Float, stdex::dextents<size_t, 1>> _tmp{_plan.size()};
    stdex::mdarray<std::complex<Float>, stdex::dextents<size_t, 1>> _signal_spectrum{_plan.size() / 2 + 1};
    stdex::mdarray<std::complex<Float>, stdex::dextents<size_t, 1>> _patch_spectrum{_plan.size() / 2 + 1};
};

template<in_vector Signal, in_vector Patch>
    requires(std::floating_point<typename Signal::value_type> and std::floating_point<typename Patch::value_type>)
auto fft_convolve(Signal signal, Patch patch)
{
    using Float = typename Signal::value_type;

    if (signal.extent(0) == 0 or patch.extent(0) == 0) {
        return stdex::mdarray<Float, stdex::dextents<size_t, 1>>{};
    }

    auto convolver = fft_convolver<Float>{signal.extent(0), patch.extent(0)};
    auto output    = stdex::mdarray<Float, stdex::dextents<size_t, 1>>{convolver.output_size()};
    convolver(signal, patch, output.to_mdspan());
    return output;
}

}  // namespace neo
