#include "convolution.hpp"

#include "neo/fft/rfft.hpp"
#include "neo/math.hpp"

namespace neo::fft
{

auto upols_convolver::filter(KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>> filter) -> void
{
    auto const K     = std::bit_ceil((filter.extent(1) - 1U) * 2U);
    auto const order = juce::roundToInt(std::log2(K));

    _fdlWritePos = 0;
    _fdl         = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>{filter.extents()};
    _filter      = std::move(filter);

    _fft = std::make_unique<juce::dsp::FFT>(order);
    _window.resize(K);
    _tmp.resize(_window.size());
    _accumlator.resize(_filter.extent(1));

    DBG("FFT-SIZE:" << _fft->getSize());
    DBG("WIN-SIZE:" << _window.size());
    DBG("ACC-SIZE:" << _accumlator.size());

    DBG("SIGNAL-EXTENT(0):" << _fdl.extent(0));
    DBG("FILTER-EXTENT(0):" << _filter.extent(0));
    DBG("SIGNAL-EXTENT(1):" << _fdl.extent(1));
    DBG("FILTER-EXTENT(1):" << _filter.extent(1));
}

auto upols_convolver::operator()(std::span<float> block) -> void
{
    jassert(block.size() * 2U == _window.size());

    auto const B = std::ssize(block);

    // Time domain input buffer
    std::shift_left(_window.begin(), _window.end(), B);
    std::copy(block.begin(), block.end(), std::prev(_window.end(), B));

    for (auto p{_fdl.extent(0) - 1U}; p > 0; --p)
    {
        for (auto i{0U}; i < _fdl.extent(1); ++i) { _fdl(p, i) = _fdl(p - 1U, i); }
    }

    // 2B-point R2C-FFT
    rfft(*_fft, _window, _tmp);

    // Copy to FDL
    for (auto i{0U}; i < _fdl.extent(1); ++i) { _fdl(0, i) = _tmp[i] / float(_fft->getSize()); }

    // DFT-spectrum additions
    std::fill(_accumlator.begin(), _accumlator.end(), 0.0F);
    for (auto p{0U}; p < _fdl.extent(0); ++p)
    {
        // auto const pidx = (p + _fdlWritePos) % _fdl.extent(0);
        for (auto i{0U}; i < _fdl.extent(1); ++i) { _accumlator[i] += _fdl(p, i) * _filter(p, i); }
    }

    // Increment fdl position
    if (++_fdlWritePos == _fdl.extent(0)) { _fdlWritePos = 0; }

    // 2B-point C2R-IFFT
    auto tmp = std::vector<float>(_window.size());
    irfft(*_fft, _accumlator, tmp);

    // Copy B samples to output
    std::copy(std::prev(tmp.end(), B), tmp.end(), block.begin());
}

static auto partition_filter(juce::AudioBuffer<float> const& buffer, int blockSize)
    -> KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>
{
    auto const windowSize = blockSize * 2;
    auto const order      = juce::roundToInt(std::log2(windowSize));
    auto const numBins    = windowSize / 2 + 1;

    auto result = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>{
        static_cast<size_t>(div_round(buffer.getNumSamples(), blockSize)),
        numBins,
    };

    auto fft    = juce::dsp::FFT{order};
    auto input  = std::vector<std::complex<float>>(size_t(windowSize));
    auto output = std::vector<std::complex<float>>(size_t(windowSize));

    for (auto f{0UL}; f < result.extent(0); ++f)
    {
        auto const idx        = static_cast<int>(f * result.extent(1));
        auto const numSamples = std::min(buffer.getNumSamples() - idx, blockSize);
        for (auto i{0}; i < numSamples; ++i) { input[static_cast<size_t>(i)] = buffer.getSample(0, idx + i); }

        std::fill(output.begin(), output.end(), 0.0F);
        fft.perform(input.data(), output.data(), false);
        for (auto b{0UL}; b < result.extent(1); ++b) { result(f, b) = output[b] / float(windowSize); }
    }

    return result;
}

auto convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter)
    -> juce::AudioBuffer<float>
{
    auto const blockSize = 512;

    auto ir = filter;
    juce_normalization(ir);

    auto convolver = upols_convolver{};
    convolver.filter(partition_filter(ir, blockSize));

    auto output = juce::AudioBuffer<float>{1, signal.getNumSamples()};
    auto block  = std::vector<float>(size_t(blockSize));

    auto const* const in = signal.getReadPointer(0);
    auto* const out      = output.getWritePointer(0);

    for (auto i{0}; i < output.getNumSamples(); i += blockSize)
    {
        auto const numSamples = std::min(output.getNumSamples() - i, blockSize);
        std::fill(block.begin(), block.end(), 0.0F);
        std::copy(std::next(in, i), std::next(in, i + numSamples), block.begin());
        convolver(block);
        std::copy(block.begin(), std::next(block.begin(), numSamples), std::next(out, i));
    }

    return output;
}

}  // namespace neo::fft
