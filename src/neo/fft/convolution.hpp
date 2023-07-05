#pragma once

#include "neo/mdspan.hpp"

#include <juce_dsp/juce_dsp.h>

#include <algorithm>
#include <vector>

namespace neo::fft
{

struct upols_convolver
{
    upols_convolver() = default;

    auto filter(KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>> filter) -> void;
    auto operator()(std::span<float> block) -> void;

private:
    std::vector<float> _window;
    std::vector<std::complex<float>> _accumlator;
    std::size_t _fdlWritePos{0};
    KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>> _fdl;
    KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>> _filter;

    std::unique_ptr<juce::dsp::FFT> _fft;
    std::vector<std::complex<float>> _tmp;
};

[[nodiscard]] auto convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter)
    -> juce::AudioBuffer<float>;

}  // namespace neo::fft
