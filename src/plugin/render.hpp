#pragma once

#include <juce_dsp/juce_dsp.h>

namespace neo
{

template<typename Processor>
auto processBlocks(Processor& processor, juce::dsp::AudioBlock<float const> const& input,
                   juce::dsp::AudioBlock<float> const& output, std::size_t blockSize, double sampleRate) -> void
{
    jassert(input.getNumChannels() == output.getNumChannels());
    jassert(input.getNumSamples() == output.getNumSamples());

    processor.prepare(juce::dsp::ProcessSpec{
        sampleRate,
        static_cast<std::uint32_t>(blockSize),
        static_cast<std::uint32_t>(input.getNumChannels()),
    });

    for (auto i{0UL}; i < output.getNumSamples(); i += blockSize)
    {
        auto const numSamples = std::min(output.getNumSamples() - i, blockSize);

        auto in  = input.getSubBlock(i, numSamples);
        auto out = output.getSubBlock(i, numSamples);
        auto ctx = juce::dsp::ProcessContextNonReplacing<float>{in, out};
        processor.process(ctx);
    }

    processor.reset();
}

}  // namespace neo
