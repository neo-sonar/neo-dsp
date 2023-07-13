#pragma once

#include "neo/convolution/container/mdspan.hpp"
#include "neo/fft.hpp"

#include <juce_dsp/juce_dsp.h>

#include <algorithm>
#include <vector>

namespace neo::fft
{

struct upols_convolver
{
    upols_convolver() = default;

    auto filter(KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> filter) -> void;
    auto operator()(std::span<float> block) -> void;

private:
    std::vector<float> _window;
    std::vector<std::complex<float>> _accumulator;
    KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>> _fdl;
    KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> _filter;
    std::size_t _fdlIndex{0};

    std::unique_ptr<rfft_radix2_plan<float>> _rfft;
    std::vector<std::complex<float>> _rfftBuf;
    std::vector<float> _irfftBuf;
};

[[nodiscard]] auto convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter,
                            float thresholdDB) -> juce::AudioBuffer<float>;

}  // namespace neo::fft
