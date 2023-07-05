#include "rfft.hpp"

namespace neo::fft
{

rfft_plan::rfft_plan(std::size_t size)
    : _fft{juce::roundToInt(std::log2(size))}
    , _in(static_cast<std::size_t>(_fft.getSize()))
    , _out(static_cast<std::size_t>(_fft.getSize()))
{
}

auto rfft_plan::operator()(std::span<float const> input, std::span<std::complex<float>> output) -> void
{
    jassert(_fft.getSize() == static_cast<int>(output.size()));

    std::fill(_in.begin(), _in.end(), 0.0F);
    std::fill(_out.begin(), _out.end(), 0.0F);

    std::copy(input.begin(), input.end(), _in.begin());
    _fft.perform(_in.data(), _out.data(), false);
    std::copy(_out.begin(), std::next(_out.begin(), _fft.getSize() / 2 + 1), output.begin());
}

auto rfft_plan::operator()(std::span<std::complex<float> const> input, std::span<float> output) -> void
{
    jassert(_fft.getSize() == static_cast<int>(output.size()));

    std::fill(_in.begin(), _in.end(), 0.0F);
    std::fill(_out.begin(), _out.end(), 0.0F);

    std::copy(input.begin(), input.end(), _in.begin());
    _fft.perform(_in.data(), _out.data(), true);
    std::transform(_out.begin(), _out.end(), output.begin(), [](auto c) { return c.real(); });
}

auto rfft(rfft_plan& plan, std::span<float const> input, std::span<std::complex<float>> output) -> void
{
    plan(input, output);
}

auto irfft(rfft_plan& plan, std::span<std::complex<float> const> input, std::span<float> output) -> void
{
    plan(input, output);
}

}  // namespace neo::fft
