#include "real_fft.hpp"

namespace neo::fft
{

rfft_plan::rfft_plan(std::size_t size)
    : _fft{juce::roundToInt(std::log2(size))}, _buf(static_cast<std::size_t>(_fft.getSize()) * 2U)
{
}

auto rfft_plan::operator()(std::span<float const> input, std::span<std::complex<float>> output) -> void
{
    jassert(_fft.getSize() == static_cast<int>(output.size()));

    std::fill(_buf.begin(), _buf.end(), 0.0F);
    std::copy(input.begin(), input.end(), _buf.begin());

    _fft.performRealOnlyForwardTransform(_buf.data());

    auto* const out = reinterpret_cast<std::complex<float>*>(_buf.data());
    std::copy(out, std::next(out, _fft.getSize() / 2 + 1), output.begin());
}

auto rfft_plan::operator()(std::span<std::complex<float> const> input, std::span<float> output) -> void
{
    jassert(_fft.getSize() == static_cast<int>(output.size()));

    auto const* in = reinterpret_cast<float const*>(input.data());
    std::fill(_buf.begin(), _buf.end(), 0.0F);
    std::copy(in, std::next(in, std::ssize(input) * 2), _buf.begin());

    _fft.performRealOnlyInverseTransform(_buf.data());

    std::copy(_buf.begin(), std::next(_buf.begin(), _fft.getSize()), output.begin());
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
