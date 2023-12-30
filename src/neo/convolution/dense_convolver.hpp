// SPDX-License-Identifier: MIT
#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/algorithm/multiply_add.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/convolution/dense_fdl.hpp>
#include <neo/convolution/overlap_add.hpp>
#include <neo/convolution/overlap_save.hpp>
#include <neo/convolution/uniform_partitioned_convolver.hpp>

namespace neo {

template<typename Complex>
struct dense_filter
{
    using value_type = Complex;

    dense_filter() = default;

    auto filter(in_matrix_of<Complex> auto filter) -> void
    {
        _filter = stdex::mdarray<Complex, stdex::dextents<size_t, 2>>{filter.extents()};
        copy(filter, _filter.to_mdspan());
    }

    auto operator()(
        in_vector_of<Complex> auto fdl,
        std::integral auto filter_index,
        inout_vector_of<Complex> auto accumulator
    ) -> void
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
    using value_type = Float;

    dense_split_filter() = default;

    auto filter(in_matrix auto filter) -> void
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

    template<in_vector InVec>
    auto operator()(split_complex<InVec> fdl, std::integral auto filter_index, inout_matrix auto accumulator) -> void
    {
        auto const subfilter = split_complex{
            stdex::submdspan(_filter.to_mdspan(), 0, filter_index, stdex::full_extent),
            stdex::submdspan(_filter.to_mdspan(), 1, filter_index, stdex::full_extent),
        };
        auto const acc = split_complex{
            stdex::submdspan(accumulator, 0, stdex::full_extent),
            stdex::submdspan(accumulator, 1, stdex::full_extent),
        };
        multiply_add(fdl, subfilter, acc, acc);
    }

private:
    stdex::mdarray<Float, stdex::dextents<size_t, 3>> _filter;
};

template<complex Complex>
using upols_convolver = uniform_partitioned_convolver<overlap_save<Complex>, dense_fdl<Complex>, dense_filter<Complex>>;

template<complex Complex>
using upola_convolver = uniform_partitioned_convolver<overlap_add<Complex>, dense_fdl<Complex>, dense_filter<Complex>>;

}  // namespace neo
