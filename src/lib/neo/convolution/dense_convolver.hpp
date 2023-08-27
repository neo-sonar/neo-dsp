#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/multiply_add.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/container/sparse_matrix.hpp>
#include <neo/convolution/dense_fdl.hpp>
#include <neo/convolution/overlap_add.hpp>
#include <neo/convolution/overlap_save.hpp>
#include <neo/convolution/uniform_partitioned_convolver.hpp>

namespace neo::fft {

template<typename Complex>
struct dense_filter
{
    using value_type = Complex;

    dense_filter() = default;

    auto filter(in_matrix auto filter) -> void { _filter = filter; }

    auto operator()(in_vector auto fdl, std::integral auto filter_index, inout_vector auto accumulator) -> void
    {
        auto const subfilter = stdex::submdspan(_filter, filter_index, stdex::full_extent);
        multiply_add(fdl, subfilter, accumulator, accumulator);
    }

private:
    stdex::mdspan<Complex const, stdex::dextents<size_t, 2>> _filter;
};

template<complex Complex>
using upols_convolver = uniform_partitioned_convolver<overlap_save<Complex>, dense_fdl<Complex>, dense_filter<Complex>>;

template<complex Complex>
using upola_convolver = uniform_partitioned_convolver<overlap_add<Complex>, dense_fdl<Complex>, dense_filter<Complex>>;

}  // namespace neo::fft
