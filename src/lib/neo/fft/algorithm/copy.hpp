#pragma once

#include <neo/fft/container/mdspan.hpp>

#include <algorithm>
#include <cstddef>
#include <span>
#include <utility>

namespace neo::fft {

inline auto copy(
    Kokkos::mdspan<float const, Kokkos::dextents<std::size_t, 2>> src,
    Kokkos::mdspan<float, Kokkos::dextents<std::size_t, 2>> dest
) -> void
{
    assert(src.extent(0) == dest.extent(0));
    assert(src.extent(1) * 2 == dest.extent(1));

    auto const numChannels = src.extent(0);
    auto const numSamples  = src.extent(1);

    for (auto ch{0UL}; ch < numChannels; ++ch) {
        auto source      = std::span{std::addressof(src(ch, 0)), numSamples};
        auto destination = std::span{std::addressof(dest(ch, numSamples)), numSamples};
        std::copy(source.begin(), source.end(), destination.begin());
    }
}

}  // namespace neo::fft
