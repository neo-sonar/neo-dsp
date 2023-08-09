#include "convolution.hpp"

#include "dsp/normalize.hpp"
#include "dsp/resample.hpp"
#include "neo/fft/convolution/uniform_partition.hpp"

namespace neo::fft {

static auto uniform_partition(juce::AudioBuffer<float> const& buffer, int blockSize)
    -> KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 3>>
{
    auto matrix = to_mdarray(buffer);
    return uniform_partition(matrix.to_mdspan(), static_cast<std::size_t>(blockSize));
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

auto dense_stereo_convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter)
    -> juce::AudioBuffer<float>
{
    auto blockSize = 512;
    auto block     = KokkosEx::mdarray<float, Kokkos::dextents<size_t, 2>>{2, size_t(blockSize)};

    auto output     = juce::AudioBuffer<float>{signal.getNumChannels(), signal.getNumSamples()};
    auto partitions = uniform_partition(filter, blockSize);

    auto convolver = stereo_upols_convolver{};
    convolver.filter(partitions);

    for (auto i{0}; i < output.getNumSamples(); i += blockSize) {
        auto const numSamples = std::min(output.getNumSamples() - i, blockSize);
        std::fill(block.data(), block.data() + block.size(), 0.0F);

        for (auto j{0}; j < numSamples; ++j) {
            block(0, static_cast<size_t>(j)) = signal.getSample(0, i + j);
            block(1, static_cast<size_t>(j)) = signal.getSample(1, i + j);
        }

        convolver(block);

        for (auto j{0}; j < numSamples; ++j) {
            output.setSample(0, i + j, block(0, static_cast<size_t>(j)));
            output.setSample(1, i + j, block(1, static_cast<size_t>(j)));
        }
    }

    return output;
}

auto sparse_convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter, float thresholdDB)
    -> juce::AudioBuffer<float>
{
    auto const blockSize = 512;

    auto output     = juce::AudioBuffer<float>{signal.getNumChannels(), signal.getNumSamples()};
    auto block      = std::vector<float>(size_t(blockSize));
    auto partitions = uniform_partition(filter, blockSize);

    for (auto ch{0}; ch < signal.getNumChannels(); ++ch) {
        auto convolver     = sparse_upols_convolver{thresholdDB};
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

}  // namespace neo::fft
