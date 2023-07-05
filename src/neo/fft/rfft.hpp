#pragma once

#include <juce_dsp/juce_dsp.h>

#include <complex>
#include <span>

namespace neo::fft
{

auto rfft(juce::dsp::FFT& fft, std::span<float const> input, std::span<std::complex<float>> output) -> void;
auto irfft(juce::dsp::FFT& fft, std::span<std::complex<float> const> input, std::span<float> output) -> void;

}  // namespace neo::fft
