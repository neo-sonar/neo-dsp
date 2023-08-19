#pragma once

#include <neo/algorithm/multiply_add.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/container/sparse_matrix.hpp>
#include <neo/fft/convolution/dense_fdl.hpp>
#include <neo/fft/convolution/overlap_add.hpp>
#include <neo/fft/convolution/overlap_save.hpp>
#include <neo/fft/convolution/uniform_partitioned_convolver.hpp>

#include <concepts>

namespace neo::fft {

template<typename Complex>
struct sparse_filter
{
    using value_type = Complex;

    sparse_filter() = default;

    auto filter(in_matrix auto filter, auto sparsity) -> void { _filter = sparse_matrix<Complex>{filter, sparsity}; }

    auto operator()(in_vector auto fdl, std::integral auto filter_index, inout_vector auto accumulator) -> void
    {
        multiply_add(fdl, _filter, filter_index, accumulator, accumulator);
    }

private:
    sparse_matrix<Complex> _filter;
};

template<std::floating_point Float, typename Complex = std::complex<Float>>
using sparse_upols_convolver
    = uniform_partitioned_convolver<overlap_save<Float>, dense_fdl<Complex>, sparse_filter<Complex>>;

template<std::floating_point Float, typename Complex = std::complex<Float>>
using sparse_upola_convolver
    = uniform_partitioned_convolver<overlap_add<Float>, dense_fdl<Complex>, sparse_filter<Complex>>;

}  // namespace neo::fft
