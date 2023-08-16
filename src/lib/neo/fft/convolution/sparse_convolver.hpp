#pragma once

#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/container/sparse_matrix.hpp>
#include <neo/fft/convolution/multiply_elements_add_columns.hpp>
#include <neo/fft/convolution/overlap_add.hpp>
#include <neo/fft/convolution/overlap_save.hpp>
#include <neo/fft/convolution/uniform_partitioned_convolver.hpp>

#include <concepts>

namespace neo::fft {

template<typename Float>
struct sparse_filter
{
    using value_type = Float;

    sparse_filter() = default;

    auto filter(in_matrix auto filter, auto sparsity) -> void
    {
        _filter = sparse_matrix<std::complex<Float>>{filter, sparsity};
    }

    auto operator()(in_matrix auto fdl, out_vector auto accumulator) -> void
    {
        multiply_elements_add_columns(fdl, _filter, accumulator);
    }

private:
    sparse_matrix<std::complex<Float>> _filter;
};

template<std::floating_point Float>
using sparse_upols_convolver = uniform_partitioned_convolver<sparse_filter<Float>, overlap_save<Float>>;

template<std::floating_point Float>
using sparse_upola_convolver = uniform_partitioned_convolver<sparse_filter<Float>, overlap_add<Float>>;

}  // namespace neo::fft
