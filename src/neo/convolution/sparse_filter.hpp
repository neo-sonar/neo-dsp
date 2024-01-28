// SPDX-License-Identifier: MIT

#pragma once

#include <neo/algorithm/multiply_add.hpp>
#include <neo/complex.hpp>
#include <neo/container/csr_matrix.hpp>
#include <neo/container/mdspan.hpp>

#include <concepts>

namespace neo::convolution {

/// \ingroup neo-convolution
template<complex Complex>
struct sparse_filter
{
    using value_type       = Complex;
    using storage_type     = csr_matrix<Complex>;
    using index_type       = typename storage_type::index_type;
    using accumulator_type = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>;

    sparse_filter() = default;

    auto filter(in_matrix_of<Complex> auto input, auto sparsity) -> void
    {
        _filter = csr_matrix<Complex>{input, sparsity};
    }

    template<in_vector_of<Complex> FdlRow, std::integral Index, inout_vector_of<Complex> Accumulator>
    auto operator()(FdlRow fdl, Index filter_index, Accumulator accumulator) -> void
    {
        multiply_add(fdl, _filter, static_cast<index_type>(filter_index), accumulator, accumulator);
    }

private:
    csr_matrix<Complex> _filter;
};

}  // namespace neo::convolution
