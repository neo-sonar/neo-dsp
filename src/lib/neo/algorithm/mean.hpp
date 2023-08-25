#pragma once

#include <neo/config.hpp>
#include <neo/container/mdspan.hpp>

#include <optional>
#include <utility>

namespace neo {

template<in_object InObj>
[[nodiscard]] constexpr auto mean(InObj x) noexcept -> std::optional<typename InObj::value_type>
{
    using Float = typename InObj::value_type;

    if (x.extent(0) == 0) {
        return std::nullopt;
    }

    auto sum = Float(0);

    if constexpr (InObj::rank() == 1) {
        for (auto i{0}; std::cmp_less(i, x.extent(0)); ++i) {
            sum += x[i];
        }
    } else {
        for (auto i{0}; std::cmp_less(i, x.extent(0)); ++i) {
            for (auto j{0}; std::cmp_less(j, x.extent(1)); ++j) {
                sum += x(i, j);
            }
        }
    }

    return sum / static_cast<Float>(x.size());
}

}  // namespace neo
