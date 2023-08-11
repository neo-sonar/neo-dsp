#pragma once

#include <neo/fft/container/mdspan.hpp>

#include <cmath>
#include <concepts>
#include <optional>
#include <type_traits>

namespace neo::fft {

template<in_vector InVec1, in_vector InVec2>
    requires(std::floating_point<typename InVec1::value_type> and std::floating_point<typename InVec2::value_type>)
auto rms_error(InVec1 signal, InVec2 reconstructed) noexcept
    -> std::optional<std::common_type_t<typename InVec1::value_type, typename InVec2::value_type>>
{
    using Float      = std::common_type_t<typename InVec1::value_type, typename InVec2::value_type>;
    using index_type = std::common_type_t<typename InVec1::index_type, typename InVec2::index_type>;

    if (signal.extents() != reconstructed.extents()) {
        return std::nullopt;
    }

    if (signal.extent(0) == 0) {
        return std::nullopt;
    }

    auto sum = Float(0);
    for (index_type i{0}; i < static_cast<index_type>(signal.extent(0)); ++i) {
        auto const diff    = signal[i] - reconstructed[i];
        auto const squared = diff * diff;
        sum += squared;
    }

    return std::sqrt(sum / static_cast<Float>(signal.size()));
}

}  // namespace neo::fft
