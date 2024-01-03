// SPDX-License-Identifier: MIT

#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/complex/complex.hpp>
#include <neo/complex/split_complex.hpp>
#include <neo/container/mdspan.hpp>

namespace neo::convolution {

template<complex Complex>
struct dense_fdl
{
    using value_type       = Complex;
    using accumulator_type = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>;

    dense_fdl() = default;

    explicit dense_fdl(stdex::dextents<size_t, 2> extents) : _fdl{extents} {}

    [[nodiscard]] auto operator[](std::integral auto index) const noexcept -> in_vector_of<Complex> auto
    {
        return stdex::submdspan(_fdl.to_mdspan(), index, stdex::full_extent);
    }

    auto insert(in_vector_of<Complex> auto input, std::integral auto index) noexcept -> void
    {
        copy(input, stdex::submdspan(_fdl.to_mdspan(), index, stdex::full_extent));
    }

private:
    stdex::mdarray<Complex, stdex::dextents<size_t, 2>> _fdl{};
};

template<typename Float>
struct dense_split_fdl
{
    using value_type       = Float;
    using accumulator_type = stdex::mdarray<Float, stdex::extents<size_t, 2, std::dynamic_extent>>;

    dense_split_fdl() = default;

    explicit dense_split_fdl(stdex::dextents<size_t, 2> extents) : _fdl{2, extents.extent(0), extents.extent(1)} {}

    [[nodiscard]] auto operator[](std::integral auto index) const noexcept
    {
        return split_complex{
            stdex::submdspan(_fdl.to_mdspan(), 0, index, stdex::full_extent),
            stdex::submdspan(_fdl.to_mdspan(), 1, index, stdex::full_extent),
        };
    }

    auto insert(in_vector auto input, std::integral auto index) noexcept -> void
    {
        auto real = stdex::submdspan(_fdl.to_mdspan(), 0, index, stdex::full_extent);
        auto imag = stdex::submdspan(_fdl.to_mdspan(), 1, index, stdex::full_extent);
        for (auto i{0}; i < static_cast<int>(input.extent(0)); ++i) {
            real[i] = static_cast<Float>(input[i].real());
            imag[i] = static_cast<Float>(input[i].imag());
        }
    }

private:
    stdex::mdarray<Float, stdex::dextents<size_t, 3>> _fdl{};
};

}  // namespace neo::convolution
