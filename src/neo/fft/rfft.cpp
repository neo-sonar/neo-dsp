#include "rfft.hpp"

namespace neo::fft
{

auto rfft(juce::dsp::FFT& fft, std::span<float const> input, std::span<std::complex<float>> output) -> void
{
    jassert(fft.getSize() == static_cast<int>(output.size()));

    auto in  = std::vector<juce::dsp::Complex<float>>(static_cast<size_t>(fft.getSize()));
    auto out = std::vector<juce::dsp::Complex<float>>(static_cast<size_t>(fft.getSize()));
    std::copy(input.begin(), input.end(), in.begin());
    fft.perform(in.data(), out.data(), false);
    std::copy(out.begin(), std::next(out.begin(), fft.getSize() / 2 + 1), output.begin());
}

auto irfft(juce::dsp::FFT& fft, std::span<std::complex<float> const> input, std::span<float> output) -> void
{
    jassert(fft.getSize() == static_cast<int>(output.size()));

    auto in  = std::vector<juce::dsp::Complex<float>>(static_cast<size_t>(fft.getSize()));
    auto out = std::vector<juce::dsp::Complex<float>>(static_cast<size_t>(fft.getSize()));
    std::copy(input.begin(), input.end(), in.begin());
    fft.perform(in.data(), out.data(), true);
    std::transform(out.begin(), out.end(), output.begin(), [](auto c) { return c.real(); });
}

}  // namespace neo::fft
