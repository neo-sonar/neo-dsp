#pragma once

#include "neo/mdspan.hpp"

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>

namespace neo::fft
{

[[nodiscard]] auto stft(juce::AudioBuffer<float> const& buffer, int windowSize)
    -> KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>;

}  // namespace neo::fft
