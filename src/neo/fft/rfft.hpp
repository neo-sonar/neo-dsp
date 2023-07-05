#pragma once

#include <juce_dsp/juce_dsp.h>

#include <complex>
#include <cstddef>
#include <span>
#include <vector>

namespace neo::fft
{

struct rfft_plan
{
    explicit rfft_plan(std::size_t size);

    [[nodiscard]] auto size() const noexcept -> std::size_t { return static_cast<std::size_t>(_fft.getSize()); }

    auto operator()(std::span<float const> input, std::span<std::complex<float>> output) -> void;
    auto operator()(std::span<std::complex<float> const> input, std::span<float> output) -> void;

private:
    juce::dsp::FFT _fft;
    std::vector<juce::dsp::Complex<float>> _in;
    std::vector<juce::dsp::Complex<float>> _out;
};

auto rfft(rfft_plan& plan, std::span<float const> input, std::span<std::complex<float>> output) -> void;
auto irfft(rfft_plan& plan, std::span<std::complex<float> const> input, std::span<float> output) -> void;

}  // namespace neo::fft
