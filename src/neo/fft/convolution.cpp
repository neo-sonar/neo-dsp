#include "convolution.hpp"

#include "neo/math.hpp"

namespace neo::fft
{

static auto shift_rows_up(Kokkos::mdspan<std::complex<float>, Kokkos::dextents<std::size_t, 2>> matrix)
{
    for (auto p{matrix.extent(0) - 1U}; p > 0; --p)
    {
        for (auto i{0U}; i < matrix.extent(1); ++i) { matrix(p, i) = matrix(p - 1U, i); }
    }
}

static auto multiply_and_accumulate(Kokkos::mdspan<std::complex<float>, Kokkos::dextents<std::size_t, 2>> lhs,
                                    Kokkos::mdspan<std::complex<float>, Kokkos::dextents<std::size_t, 2>> rhs,
                                    std::span<std::complex<float>> accumulator)
{
    jassert(lhs.extents() == rhs.extents());
    jassert(lhs.extent(1) > 0);

    // First loop, so we don't need to clear the accumulator from previous iteration
    for (auto i{0U}; i < lhs.extent(1); ++i) { accumulator[i] = lhs(0, i) * rhs(0, i); }

    for (auto p{1U}; p < lhs.extent(0); ++p)
    {
        for (auto i{0U}; i < lhs.extent(1); ++i) { accumulator[i] += lhs(p, i) * rhs(p, i); }
    }
}

auto upols_convolver::filter(KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>> filter) -> void
{
    auto const K = std::bit_ceil((filter.extent(1) - 1U) * 2U);

    _fdl    = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>{filter.extents()};
    _filter = std::move(filter);

    _rfft = std::make_unique<rfft_plan>(K);
    _window.resize(K);
    _rfftBuf.resize(_window.size());
    _irfftBuf.resize(_window.size());
    _accumulator.resize(_filter.extent(1));
}

auto upols_convolver::operator()(std::span<float> block) -> void
{
    jassert(block.size() * 2U == _window.size());

    auto const blockSize = std::ssize(block);

    // Time domain input buffer
    std::shift_left(_window.begin(), _window.end(), blockSize);
    std::copy(block.begin(), block.end(), std::prev(_window.end(), blockSize));

    // All contents (DFT spectra) in the FDL are shifted up by one slot.
    shift_rows_up(_fdl);

    // 2B-point R2C-FFT
    rfft(*_rfft, _window, _rfftBuf);

    // Copy to FDL
    for (auto i{0U}; i < _fdl.extent(1); ++i) { _fdl(0, i) = _rfftBuf[i] / float(_rfft->size()); }

    // DFT-spectrum additions
    multiply_and_accumulate(_fdl, _filter, _accumulator);

    // 2B-point C2R-IFFT
    irfft(*_rfft, _accumulator, _irfftBuf);

    // Copy blockSize samples to output
    std::copy(std::prev(_irfftBuf.end(), blockSize), _irfftBuf.end(), block.begin());
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

    auto rfft   = rfft_plan{static_cast<std::size_t>(windowSize)};
    auto input  = std::vector<float>(size_t(windowSize));
    auto output = std::vector<std::complex<float>>(size_t(windowSize));

    for (auto f{0UL}; f < result.extent(0); ++f)
    {
        auto const idx        = static_cast<int>(f * result.extent(1));
        auto const numSamples = std::min(buffer.getNumSamples() - idx, blockSize);
        for (auto i{0}; i < numSamples; ++i) { input[static_cast<size_t>(i)] = buffer.getSample(0, idx + i); }

        std::fill(output.begin(), output.end(), 0.0F);
        rfft(input, output);
        for (auto b{0UL}; b < result.extent(1); ++b) { result(f, b) = output[b] / float(windowSize); }
    }

    return result;
}

auto convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter)
    -> juce::AudioBuffer<float>
{
    auto const blockSize = 512;

    auto convolver = upols_convolver{};
    convolver.filter(partition_filter(filter, blockSize));

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
