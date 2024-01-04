// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_graphics/juce_graphics.h>

namespace neo {

[[nodiscard]] auto powerSpectrumImage(
    stdex::mdspan<std::complex<float> const, stdex::dextents<size_t, 2>> frames,
    std::function<float(std::size_t)> const& weighting,
    float threshold
) -> juce::Image;

[[nodiscard]] auto powerHistogram(
    stdex::mdspan<std::complex<float> const, stdex::dextents<size_t, 2>> spectogram,
    std::function<float(std::size_t)> const& weighting
) -> std::vector<int>;

[[nodiscard]] auto powerHistogramImage(
    stdex::mdspan<std::complex<float> const, stdex::dextents<size_t, 2>> spectogram,
    std::function<float(std::size_t)> const& weighting
) -> juce::Image;

}  // namespace neo
