// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/algorithm/multiply_add.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/type_traits/value_type_t.hpp>

namespace neo::convolution {

template<typename Complex>
struct dense_filter
{
    using value_type       = Complex;
    using accumulator_type = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>;

    dense_filter() = default;

    auto filter(in_matrix_of<Complex> auto input) -> void
    {
        _filter = stdex::mdarray<Complex, stdex::dextents<size_t, 2>>{input.extents()};
        copy(input, _filter.to_mdspan());
    }

    template<in_vector_of<Complex> FdlRow, std::integral Index, inout_vector_of<Complex> Accumulator>
    auto operator()(FdlRow fdl, Index filter_index, Accumulator accumulator) -> void
    {
        auto const subfilter = stdex::submdspan(_filter.to_mdspan(), filter_index, stdex::full_extent);
        multiply_add(fdl, subfilter, accumulator, accumulator);
    }

private:
    stdex::mdarray<Complex, stdex::dextents<size_t, 2>> _filter;
};

template<typename Float>
struct dense_split_filter
{
    using value_type       = Float;
    using accumulator_type = stdex::mdarray<Float, stdex::extents<size_t, 2, std::dynamic_extent>>;

    dense_split_filter() = default;

    template<in_matrix Filter>
        requires complex<value_type_t<Filter>>
    auto filter(Filter filter) -> void
    {
        _filter    = stdex::mdarray<Float, stdex::dextents<size_t, 3>>{2, filter.extent(0), filter.extent(1)};
        auto reals = stdex::submdspan(_filter.to_mdspan(), 0, stdex::full_extent, stdex::full_extent);
        auto imags = stdex::submdspan(_filter.to_mdspan(), 1, stdex::full_extent, stdex::full_extent);

        for (auto i{0}; std::cmp_less(i, filter.extent(0)); ++i) {
            for (auto j{0}; std::cmp_less(j, filter.extent(1)); ++j) {
                reals(i, j) = static_cast<Float>(filter(i, j).real());
                imags(i, j) = static_cast<Float>(filter(i, j).imag());
            }
        }
    }

    template<in_vector InVec, std::integral Index, inout_matrix_of<Float> Accumulator>
    auto operator()(split_complex<InVec> fdl, Index filter_index, Accumulator accumulator) -> void
    {
        auto const subfilter = split_complex{
            stdex::submdspan(_filter.to_mdspan(), 0, filter_index, stdex::full_extent),
            stdex::submdspan(_filter.to_mdspan(), 1, filter_index, stdex::full_extent),
        };
        auto const out = split_complex{
            stdex::submdspan(accumulator, 0, stdex::full_extent),
            stdex::submdspan(accumulator, 1, stdex::full_extent),
        };
        multiply_add(fdl, subfilter, out, out);
    }

private:
    stdex::mdarray<Float, stdex::dextents<size_t, 3>> _filter;
};

}  // namespace neo::convolution
