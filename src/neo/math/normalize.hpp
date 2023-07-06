#pragma once

#include <juce_audio_basics/juce_audio_basics.h>

#include <concepts>
#include <span>

namespace neo
{

// normalized_sample = sample / max(abs(buffer))
auto peak_normalization(std::span<float> buffer) -> void;

// normalized_sample = sample / sqrt(mean(buffer^2))
auto rms_normalization(std::span<float> buffer) -> void;

auto juce_normalization(juce::AudioBuffer<float>& buf) -> void;

template<std::integral T>
[[nodiscard]] constexpr auto div_round(T x, T y) noexcept -> T
{
    return (x + y - 1) / y;
}

}  // namespace neo
