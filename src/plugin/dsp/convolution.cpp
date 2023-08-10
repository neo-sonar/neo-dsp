#include "convolution.hpp"

#include <neo/fft/math/a_weighting.hpp>
#include <neo/fft/math/decibel.hpp>
#include <neo/fft/math/next_power_of_two.hpp>

#include "dsp/normalize.hpp"
#include "dsp/resample.hpp"
#include "neo/fft/convolution/uniform_partition.hpp"

namespace neo::fft {

static auto uniform_partition(juce::AudioBuffer<float> const& buffer, int blockSize)
    -> KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 3>>
{
    auto matrix = to_mdarray(buffer);
    return uniform_partition<float>(matrix.to_mdspan(), static_cast<std::size_t>(blockSize));
}

auto dense_convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter)
    -> juce::AudioBuffer<float>
{
    auto const blockSize = 512;

    auto output     = juce::AudioBuffer<float>{signal.getNumChannels(), signal.getNumSamples()};
    auto block      = std::vector<float>(size_t(blockSize));
    auto partitions = uniform_partition(filter, blockSize);

    for (auto ch{0}; ch < signal.getNumChannels(); ++ch) {
        auto convolver     = upols_convolver{};
        auto const channel = static_cast<size_t>(ch);
        auto const full    = Kokkos::full_extent;
        convolver.filter(KokkosEx::submdspan(partitions.to_mdspan(), channel, full, full));

        auto const* const in = signal.getReadPointer(ch);
        auto* const out      = output.getWritePointer(ch);

        for (auto i{0}; i < output.getNumSamples(); i += blockSize) {
            auto const numSamples = std::min(output.getNumSamples() - i, blockSize);
            std::fill(block.begin(), block.end(), 0.0F);
            std::copy(std::next(in, i), std::next(in, i + numSamples), block.begin());
            convolver(block);
            std::copy(block.begin(), std::next(block.begin(), numSamples), std::next(out, i));
        }
    }

    return output;
}

[[nodiscard]] static auto
normalization_factor(Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> filter) -> float
{
    auto maxPower = 0.0F;
    for (auto p{0UL}; p < filter.extent(0); ++p) {
        auto const partition = KokkosEx::submdspan(filter, p, Kokkos::full_extent);
        for (auto bin{0UL}; bin < filter.extent(1); ++bin) {
            auto const amplitude = std::abs(partition(bin));
            maxPower             = std::max(maxPower, amplitude * amplitude);
        }
    }
    return 1.0F / maxPower;
}

auto sparse_convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter, float thresholdDB)
    -> juce::AudioBuffer<float>
{
    auto const blockSize = 512;

    auto output     = juce::AudioBuffer<float>{signal.getNumChannels(), signal.getNumSamples()};
    auto block      = std::vector<float>(size_t(blockSize));
    auto partitions = uniform_partition(filter, blockSize);

    auto const K = next_power_of_two((partitions.extent(2) - 1U) * 2U);

    auto const weights = [K, bins = partitions.extent(2)] {
        auto w = std::vector<float>(bins);
        for (auto i{0U}; i < w.size(); ++i) {
            auto const frequency = frequency_for_bin<float>(K, i, 44'100.0);
            auto const weight    = frequency > 0.0F ? neo::fft::a_weighting(frequency) : 0.0F;

            w[i] = weight;
        }
        return w;
    }();

    for (auto ch{0}; ch < signal.getNumChannels(); ++ch) {
        auto convolver               = sparse_upols_convolver{thresholdDB};
        auto const channel           = static_cast<size_t>(ch);
        auto const full              = Kokkos::full_extent;
        auto const channelPartitions = KokkosEx::submdspan(partitions.to_mdspan(), channel, full, full);

        auto const scale = normalization_factor(channelPartitions);

        auto const isAboveThreshold = [thresholdDB, scale, &weights](auto /*row*/, auto col, auto bin) {
            auto const gain  = std::abs(bin);
            auto const power = gain * gain;
            auto const dB    = to_decibels(power * scale) + weights[col];
            return dB > thresholdDB;
        };

        convolver.filter(channelPartitions, isAboveThreshold);

        auto const* const in = signal.getReadPointer(ch);
        auto* const out      = output.getWritePointer(ch);

        for (auto i{0}; i < output.getNumSamples(); i += blockSize) {
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
