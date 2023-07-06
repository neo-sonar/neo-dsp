#pragma once

#include "neo/fft/rfft.hpp"
#include "neo/mdspan.hpp"

#include <juce_dsp/juce_dsp.h>

#include <algorithm>
#include <vector>

namespace neo::fft
{

struct upols_convolver
{
    upols_convolver() = default;

    auto filter(KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>> filter) -> void;
    auto operator()(std::span<float> block) -> void;

private:
    std::vector<float> _window;
    std::vector<std::complex<float>> _accumulator;
    KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>> _fdl;
    KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>> _filter;

    std::unique_ptr<rfft_plan> _rfft;
    std::vector<std::complex<float>> _rfftBuf;
    std::vector<float> _irfftBuf;
};

[[nodiscard]] auto convolve(juce::AudioBuffer<float> const& signal, juce::AudioBuffer<float> const& filter)
    -> juce::AudioBuffer<float>;

struct juce_convolver
{
    explicit juce_convolver(juce::File impulse) : _impulse{std::move(impulse)} {}

    auto prepare(juce::dsp::ProcessSpec const& spec) -> void
    {
        auto const trim      = juce::dsp::Convolution::Trim::no;
        auto const stereo    = juce::dsp::Convolution::Stereo::no;
        auto const normalize = juce::dsp::Convolution::Normalise::no;

        _convolver.prepare(spec);
        _convolver.loadImpulseResponse(_impulse, stereo, trim, 0, normalize);

        // impulse is loaded on background thread, may not have loaded fast enough in unit-tests
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }

    auto reset() -> void { _convolver.reset(); }

    template<typename Context>
    auto process(Context const& context) -> void
    {
        _convolver.process(context);
    }

private:
    juce::File _impulse;
    juce::dsp::ConvolutionMessageQueue _queue;
    juce::dsp::Convolution _convolver{juce::dsp::Convolution::Latency{0}, _queue};
};

}  // namespace neo::fft
