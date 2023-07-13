#include "convolution.hpp"

#include "dsp/normalize.hpp"

namespace neo::fft
{

static auto multiply_and_accumulate_row(std::span<std::complex<float> const> lhs,
                                        std::span<std::complex<float> const> rhs,
                                        std::span<std::complex<float>> accumulator)
{
    auto* NEO_FFT_RESTRICT acc         = accumulator.data();
    auto const* NEO_FFT_RESTRICT left  = lhs.data();
    auto const* NEO_FFT_RESTRICT right = rhs.data();
    for (decltype(lhs.size()) i{0}; i < lhs.size(); ++i) { acc[i] += left[i] * right[i]; }
}

static auto multiply_and_accumulate(Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<std::size_t, 2>> lhs,
                                    Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<std::size_t, 2>> rhs,
                                    std::span<std::complex<float>> accumulator, std::size_t shift)
{
    jassert(lhs.extents() == rhs.extents());
    jassert(lhs.extent(1) > 0);
    jassert(shift < lhs.extent(0));

    auto getRow = [](auto const& matrix, size_t row) {
        return std::span<std::complex<float> const>{std::addressof(matrix(row, 0)), matrix.extent(1)};
    };

    // First loop, so we don't need to clear the accumulator from previous iteration
    multiply_and_accumulate_row(getRow(lhs, 0), getRow(rhs, shift), accumulator);

    for (auto row{1U}; row <= shift; ++row)
    {
        multiply_and_accumulate_row(getRow(lhs, row), getRow(rhs, shift - row), accumulator);
    }

    for (auto row{shift + 1}; row < lhs.extent(0); ++row)
    {
        auto const offset    = row - shift;
        auto const offsetRow = lhs.extent(0) - offset - 1;
        multiply_and_accumulate_row(getRow(lhs, row), getRow(rhs, offsetRow), accumulator);
    }
}

static auto multiply_and_accumulate(Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<std::size_t, 2>> lhs,
                                    sparse_matrix<std::complex<float>> const& rhs,
                                    std::span<std::complex<float>> accumulator, std::size_t shift)
{
    jassert(lhs.extent(0) == rhs.rows());
    jassert(lhs.extent(1) == rhs.columns());
    jassert(lhs.extent(1) > 0);
    jassert(shift < lhs.extent(0));

    schur_product_accumulate_columnwise(lhs, rhs, accumulator, shift);
}

auto upols_convolver::filter(KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> filter) -> void
{
    auto const K = std::bit_ceil((filter.extent(1) - 1U) * 2U);

    _fdl    = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>{filter.extents()};
    _filter = filter;

    _rfft = std::make_unique<rfft_radix2_plan<float>>(ilog2(K));
    _window.resize(K);
    _rfftBuf.resize(_window.size());
    _irfftBuf.resize(_window.size());
    _accumulator.resize(_filter.extent(1));

    _fdlIndex = 0;
}

auto upols_convolver::operator()(std::span<float> block) -> void
{
    jassert(block.size() * 2U == _window.size());

    auto const blockSize = std::ssize(block);

    // Time domain input buffer
    std::shift_left(_window.begin(), _window.end(), blockSize);
    std::copy(block.begin(), block.end(), std::prev(_window.end(), blockSize));

    // 2B-point R2C-FFT
    std::invoke(*_rfft, _window, _rfftBuf);

    // Copy to FDL
    for (auto i{0U}; i < _fdl.extent(1); ++i) { _fdl(_fdlIndex, i) = _rfftBuf[i] / float(_rfft->size()); }

    // DFT-spectrum additions
    std::fill(_accumulator.begin(), _accumulator.end(), 0.0F);
    multiply_and_accumulate(_fdl, _filter, _accumulator, _fdlIndex);

    // All contents (DFT spectra) in the FDL are shifted up by one slot.
    ++_fdlIndex;
    if (_fdlIndex == _fdl.extent(0)) { _fdlIndex = 0; }

    // 2B-point C2R-IFFT
    std::invoke(*_rfft, _accumulator, _irfftBuf);

    // Copy blockSize samples to output
    std::copy(std::prev(_irfftBuf.end(), blockSize), _irfftBuf.end(), block.begin());
}

auto sparse_upols_convolver::filter(KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> filter)
    -> void
{
    auto const K = std::bit_ceil((filter.extent(1) - 1U) * 2U);

    _fdl    = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>{filter.extents()};
    _filter = sparse_matrix<std::complex<float>>{filter, [](auto) { return true; }};

    _rfft = std::make_unique<rfft_radix2_plan<float>>(ilog2(K));
    _window.resize(K);
    _rfftBuf.resize(_window.size());
    _irfftBuf.resize(_window.size());
    _accumulator.resize(_filter.columns());

    _fdlIndex = 0;
}

auto sparse_upols_convolver::operator()(std::span<float> block) -> void
{
    jassert(block.size() * 2U == _window.size());

    auto const blockSize = std::ssize(block);

    // Time domain input buffer
    std::shift_left(_window.begin(), _window.end(), blockSize);
    std::copy(block.begin(), block.end(), std::prev(_window.end(), blockSize));

    // 2B-point R2C-FFT
    std::invoke(*_rfft, _window, _rfftBuf);

    // Copy to FDL
    for (auto i{0U}; i < _fdl.extent(1); ++i) { _fdl(_fdlIndex, i) = _rfftBuf[i] / float(_rfft->size()); }

    // DFT-spectrum additions
    std::fill(_accumulator.begin(), _accumulator.end(), 0.0F);
    multiply_and_accumulate(_fdl, _filter, _accumulator, _fdlIndex);

    // All contents (DFT spectra) in the FDL are shifted up by one slot.
    ++_fdlIndex;
    if (_fdlIndex == _fdl.extent(0)) { _fdlIndex = 0; }

    // 2B-point C2R-IFFT
    std::invoke(*_rfft, _accumulator, _irfftBuf);

    // Copy blockSize samples to output
    std::copy(std::prev(_irfftBuf.end(), blockSize), _irfftBuf.end(), block.begin());
}

static auto partition_filter(juce::AudioBuffer<float> const& buffer, int blockSize)
    -> KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 3>>
{
    auto const windowSize = blockSize * 2;
    auto const numBins    = windowSize / 2 + 1;

    auto result = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 3>>{
        static_cast<std::size_t>(buffer.getNumChannels()),
        static_cast<std::size_t>(div_round(buffer.getNumSamples(), blockSize)),
        numBins,
    };

    auto rfft   = rfft_radix2_plan<float>{ilog2(static_cast<size_t>(windowSize))};
    auto input  = std::vector<float>(size_t(windowSize));
    auto output = std::vector<std::complex<float>>(size_t(windowSize));

    for (auto channel{0UL}; channel < result.extent(0); ++channel)
    {

        for (auto partition{0UL}; partition < result.extent(1); ++partition)
        {
            auto const idx        = static_cast<int>(partition * result.extent(2));
            auto const numSamples = std::min(buffer.getNumSamples() - idx, blockSize);
            std::fill(input.begin(), input.end(), 0.0F);
            for (auto i{0}; i < numSamples; ++i)
            {
                input[static_cast<size_t>(i)] = buffer.getSample(static_cast<int>(channel), idx + i);
            }

            std::fill(output.begin(), output.end(), 0.0F);
            rfft(input, output);
            for (auto bin{0UL}; bin < result.extent(2); ++bin)
            {
                result(channel, partition, bin) = output[bin] / float(windowSize);
            }
        }
    }

    return result;
}

static auto normalization_factor(Kokkos::mdspan<std::complex<float>, Kokkos::dextents<size_t, 3>> filter) -> float
{
    auto maxGain = 0.0F;

    for (auto ch{0UL}; ch < filter.extent(0); ++ch)
    {
        for (auto partition{0UL}; partition < filter.extent(1); ++partition)
        {
            for (auto bin{0UL}; bin < filter.extent(2); ++bin)
            {
                maxGain = std::max(maxGain, std::abs(filter(ch, partition, bin)));
            }
        }
    }

    return 1.0F / maxGain;
}

static auto clamp_coefficients_to_zero(Kokkos::mdspan<std::complex<float>, Kokkos::dextents<size_t, 3>> filter,
                                       float thresholdDB) -> void
{
    auto const factor = normalization_factor(filter);

    auto counter = 0;
    for (auto ch{0UL}; ch < filter.extent(0); ++ch)
    {
        for (auto partition{0UL}; partition < filter.extent(1); ++partition)
        {
            for (auto bin{0UL}; bin < filter.extent(2); ++bin)
            {
                auto const amplitude = std::abs(filter(ch, partition, bin)) * factor;
                auto const dB        = juce::Decibels::gainToDecibels(amplitude);
                if (dB < thresholdDB)
                {
                    filter(ch, partition, bin) = 0.0F;
                    ++counter;
                }
            }
        }
    }

    auto const percentage = double(counter) / double(filter.size()) * 100.0;
    std::cout << "Clamp(" << thresholdDB << "): " << counter << " (" << percentage << " %)\n";
}

auto convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter, float thresholdDB)
    -> juce::AudioBuffer<float>
{
    auto const blockSize = 512;

    auto output = juce::AudioBuffer<float>{signal.getNumChannels(), signal.getNumSamples()};
    auto block  = std::vector<float>(size_t(blockSize));

    auto partitions = partition_filter(filter, blockSize);
    clamp_coefficients_to_zero(partitions, thresholdDB);

    for (auto ch{0}; ch < signal.getNumChannels(); ++ch)
    {
        auto convolver     = sparse_upols_convolver{};
        auto const channel = static_cast<size_t>(ch);
        auto const full    = Kokkos::full_extent;
        convolver.filter(KokkosEx::submdspan(partitions.to_mdspan(), channel, full, full));

        auto const* const in = signal.getReadPointer(ch);
        auto* const out      = output.getWritePointer(ch);

        for (auto i{0}; i < output.getNumSamples(); i += blockSize)
        {
            auto const numSamples = std::min(output.getNumSamples() - i, blockSize);
            std::fill(block.begin(), block.end(), 0.0F);
            std::copy(std::next(in, i), std::next(in, i + numSamples), block.begin());
            convolver(block);
            std::copy(block.begin(), std::next(block.begin(), numSamples), std::next(out, i));
        }
    }

    return output;
}

}  // namespace neo::fft
