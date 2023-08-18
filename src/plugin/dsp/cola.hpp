#pragma once

#include <juce_dsp/juce_dsp.h>

namespace neo {

template<typename Processor>
struct COLA
{
    using SampleType = typename Processor::SampleType;

    explicit COLA(int frameSizeAsPowerOf2, int hopSizeDividerAsPowerOf2 = 1);

    auto prepare(juce::dsp::ProcessSpec const& spec) -> void;

    template<typename ProcessContext>
    auto process(ProcessContext const& context) -> void;

    auto reset() -> void;

    [[nodiscard]] auto processor() -> Processor&;
    [[nodiscard]] auto processor() const -> Processor const&;

private:
    auto createWindow() -> void;
    auto writeBackFrame(int numChannels) -> void;

    Processor _processor;

    juce::AudioBuffer<SampleType> _frame{};
    juce::AudioBuffer<SampleType> _unusedInput{};
    juce::AudioBuffer<SampleType> _output{};

    std::vector<SampleType> _window{};

    int _frameSize;
    int _hopSize;

    int _outputOffset{0};
    int _unusedInputCount{0};
};

template<typename Processor>
COLA<Processor>::COLA(int frameSizeAsPowerOf2, int hopSizeDividerAsPowerOf2)
    : _frameSize(1 << frameSizeAsPowerOf2)
    , _hopSize(_frameSize >> hopSizeDividerAsPowerOf2)
{}

template<typename Processor>
void COLA<Processor>::prepare(juce::dsp::ProcessSpec const& spec)
{
    _window = std::vector<SampleType>((size_t)_frameSize, (SampleType)0);
    createWindow();

    auto const bufferSize = (int)spec.maximumBlockSize;

    _unusedInput.setSize((int)spec.numChannels, _frameSize - 1);
    _frame.setSize((int)spec.numChannels, _frameSize);

    auto const k = (int)std::floor(1.0F + (SampleType(bufferSize - 1)) / (SampleType)_hopSize);
    int const m  = k * _hopSize + (_frameSize - _hopSize);

    _output.setSize((int)spec.numChannels, m + bufferSize - 1);
    _outputOffset     = _frameSize - 1;
    _unusedInputCount = 0;

    _processor.prepare({spec.sampleRate, (juce::uint32)_frameSize, spec.numChannels});
}

template<typename Processor>
void COLA<Processor>::reset()
{
    _unusedInput.clear();
    _frame.clear();
    _output.clear();

    _outputOffset     = _frameSize - 1;
    _unusedInputCount = 0;

    _processor.reset();
}

template<typename Processor>
auto COLA<Processor>::processor() -> Processor&
{
    return _processor;
}

template<typename Processor>
auto COLA<Processor>::processor() const -> Processor const&
{
    return _processor;
}

template<typename Processor>
template<typename ProcessContext>
void COLA<Processor>::process(ProcessContext const& context)
{
    auto inBlock  = context.getInputBlock();
    auto outBlock = context.getOutputBlock();

    jassert(inBlock.getNumChannels() == outBlock.getNumChannels());
    jassert(inBlock.getNumSamples() == outBlock.getNumSamples());

    auto const numSamples  = static_cast<int>(inBlock.getNumSamples());
    auto const numChannels = static_cast<int>(inBlock.getNumChannels());
    auto const l           = static_cast<int>(numSamples);

    auto const initialUnusedCount  = _unusedInputCount;
    auto notYetUsedAudioDataOffset = 0;
    auto usedSamples               = 0;

    // we've got some left overs, so let's use them together with the new samples
    while (_unusedInputCount > 0 && _unusedInputCount + l >= _frameSize) {
        // copy not yet used data into fftInOut buffer (with windowing)
        for (auto ch = 0; ch < numChannels; ++ch) {
            juce::FloatVectorOperations::multiply(
                _frame.getWritePointer(ch),
                _unusedInput.getReadPointer(ch, notYetUsedAudioDataOffset),
                _window.data(),
                _unusedInputCount
            );

            // fill up fftInOut buffer with new data (with windowing)
            juce::FloatVectorOperations::multiply(
                _frame.getWritePointer(ch, _unusedInputCount),
                inBlock.getChannelPointer((size_t)ch),
                _window.data() + _unusedInputCount,
                _frameSize - _unusedInputCount
            );
        }

        // process frame and buffer output
        _processor.process(_frame);
        writeBackFrame(numChannels);

        notYetUsedAudioDataOffset += _hopSize;
        _unusedInputCount -= _hopSize;
    }

    if (_unusedInputCount > 0) {
        // not enough new input samples to use all of the previous data
        for (auto ch = 0; ch < numChannels; ++ch) {
            std::copy_n(
                _unusedInput.getReadPointer(ch, initialUnusedCount - _unusedInputCount),
                _unusedInputCount,
                _unusedInput.getWritePointer(ch)
            );
            std::copy_n(
                std::next(inBlock.getChannelPointer((size_t)ch), usedSamples),
                l,
                _unusedInput.getWritePointer(ch, _unusedInputCount)
            );
        }
        _unusedInputCount += l;
    } else {
        // all of the previous data used
        auto dataOffset = -_unusedInputCount;

        while (l - dataOffset >= _frameSize) {
            for (auto ch = 0; ch < numChannels; ++ch) {
                juce::FloatVectorOperations::multiply(
                    _frame.getWritePointer(ch),
                    std::next(inBlock.getChannelPointer((size_t)ch), dataOffset),
                    _window.data(),
                    _frameSize
                );
            }
            _processor.process(_frame);
            writeBackFrame(numChannels);

            dataOffset += _hopSize;
        }

        auto const remainingSamples = l - dataOffset;
        if (remainingSamples > 0) {
            for (auto ch = 0; ch < numChannels; ++ch) {
                std::copy_n(
                    std::next(inBlock.getChannelPointer((size_t)ch), dataOffset),
                    l - dataOffset,
                    _unusedInput.getWritePointer(ch)
                );
            }
        }
        _unusedInputCount = remainingSamples;
    }

    // return processed samples from _output
    auto const shiftStart = l;
    auto shiftL           = _outputOffset + _frameSize - _hopSize - l;

    auto const tooMuch = shiftStart + shiftL - _output.getNumSamples();
    if (tooMuch > 0) {
        shiftL -= tooMuch;
    }

    for (auto ch = 0; ch < numChannels; ++ch) {
        std::copy_n(_output.getReadPointer(ch), l, outBlock.getChannelPointer((size_t)ch));
        std::copy_n(_output.getReadPointer(ch, shiftStart), shiftL, _output.getWritePointer(ch));
    }

    _outputOffset -= l;
}

template<typename Processor>
void COLA<Processor>::createWindow()
{
    auto type = juce::dsp::WindowingFunction<SampleType>::rectangular;
    juce::dsp::WindowingFunction<SampleType>::fillWindowingTables(_window.data(), (size_t)_frameSize, type, false);

    auto const hopSizeCompensateFactor
        = (SampleType)1 / ((SampleType)_frameSize / (SampleType)_hopSize / (SampleType)2);
    juce::FloatVectorOperations::multiply(_window.data(), hopSizeCompensateFactor, _frameSize);
}

template<typename Processor>
void COLA<Processor>::writeBackFrame(int numChannels)
{
    for (int ch = 0; ch < numChannels; ++ch) {
        _output.addFrom(ch, _outputOffset, _frame, ch, 0, _frameSize - _hopSize);
        _output.copyFrom(ch, _outputOffset + _frameSize - _hopSize, _frame, ch, _frameSize - _hopSize, _hopSize);
    }
    _outputOffset += _hopSize;
}

}  // namespace neo
