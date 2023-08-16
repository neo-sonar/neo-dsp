#pragma once

#include <neo/config.hpp>

#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/transform/direction.hpp>

#include <numbers>

namespace neo::fft {

template<inout_vector OutVec>
auto fill_radix2_twiddles(OutVec table, direction dir = direction::forward) -> void
{
    using Complex = typename OutVec::value_type;
    using Float   = typename Complex::value_type;

    auto const tableSize = table.size();
    auto const fftSize   = tableSize * 2ULL;
    auto const sign      = dir == direction::forward ? Float(-1) : Float(1);
    auto const twoPi     = static_cast<Float>(std::numbers::pi * 2.0);

    for (std::size_t i = 0; i < tableSize; ++i) {
        auto const angle = sign * twoPi * Float(i) / Float(fftSize);
        table[i]         = std::polar(Float(1), angle);
    }
}

template<typename Complex>
auto make_radix2_twiddles(std::size_t size, direction dir = direction::forward)
{
    auto table = stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>>{size / 2U};
    fill_radix2_twiddles(table.to_mdspan(), dir);
    return table;
}

template<typename Complex, std::size_t Size>
auto make_radix2_twiddles(direction dir = direction::forward)
{
    auto table = stdex::mdarray<Complex, stdex::extents<std::size_t, Size / 2>>{};
    fill_radix2_twiddles(table.to_mdspan(), dir);
    return table;
}

}  // namespace neo::fft
