#pragma once

#include <neo/container/mdspan.hpp>

#include <cmath>
#include <concepts>
#include <numbers>

namespace neo {

template<std::floating_point Float>
struct hann_window
{
    explicit hann_window(std::integral auto size) noexcept : _size{size} {}

    [[nodiscard]] auto operator()(std::integral auto index) const noexcept -> Float
    {
        auto const n     = static_cast<Float>(_size);
        auto const twoPi = static_cast<Float>(std::numbers::pi) * Float(2);
        return Float(0.5) * (Float(1) - std::cos(twoPi * static_cast<Float>(index) / (n - Float(1))));
    }

private:
    std::size_t _size;
};

auto fill_window(inout_vector auto vec, auto window)
{
    for (auto i{0}; std::cmp_less(i, vec.extent(0)); ++i) {
        vec(i) = window(i);
    }
}

template<std::floating_point Float>
[[nodiscard]] auto generate_window(std::size_t length) -> KokkosEx::mdarray<Float, Kokkos::dextents<std::size_t, 1>>
{
    auto buffer = KokkosEx::mdarray<Float, Kokkos::dextents<std::size_t, 1>>{length};
    fill_window(buffer.to_mdspan(), hann_window<Float>{length});
    return buffer;
}

}  // namespace neo
