#pragma once

#include <neo/fft/container/mdspan.hpp>

#include <cmath>
#include <concepts>
#include <type_traits>
#include <utility>

namespace neo::fft {

template<in_vector Signal, in_vector Reconstructed>
    requires(std::floating_point<typename Signal::value_type> and std::floating_point<typename Reconstructed::value_type>)
auto rms_error(Signal signal, Reconstructed reconstructed) noexcept
    -> std::common_type_t<typename Signal::value_type, typename Reconstructed::value_type>
{
    using Float = std::common_type_t<typename Signal::value_type, typename Reconstructed::value_type>;
    using Index = std::common_type_t<typename Signal::index_type, typename Reconstructed::index_type>;

    auto sum = Float(0);
    for (Index i{0}; std::cmp_less(i, signal.extent(0)); ++i) {
        auto const diff    = signal[i] - reconstructed[i];
        auto const squared = diff * diff;
        sum += squared;
    }

    return std::sqrt(sum / static_cast<Float>(signal.extent(0)));
}

}  // namespace neo::fft
