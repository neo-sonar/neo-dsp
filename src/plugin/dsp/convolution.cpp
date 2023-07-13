#include "convolution.hpp"

#include "dsp/normalize.hpp"

namespace neo::fft
{

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
