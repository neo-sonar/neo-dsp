// SPDX-License-Identifier: MIT

#pragma once

#include <neo/algorithm/multiply_add.hpp>
#include <neo/complex.hpp>
#include <neo/container/csr_matrix.hpp>
#include <neo/container/mdspan.hpp>

#include <concepts>

namespace neo::convolution {

template<complex Complex>
struct sparse_filter
{
    using value_type       = Complex;
    using accumulator_type = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>;

    sparse_filter() = default;

    auto filter(in_matrix_of<Complex> auto filter, auto sparsity) -> void
    {
        _filter = csr_matrix<Complex>{filter, sparsity};
    }

    auto operator()(
        in_vector_of<Complex> auto fdl,
        std::integral auto filter_index,
        inout_vector_of<Complex> auto accumulator
    ) -> void
    {
        multiply_add(fdl, _filter, filter_index, accumulator, accumulator);
    }

private:
    csr_matrix<Complex> _filter;
};

}  // namespace neo::convolution
