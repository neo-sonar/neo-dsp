#pragma once

#include <neo/container/mdspan.hpp>
#include <neo/container/sparse_matrix.hpp>
#include <neo/fft/convolution/multiply_elements_add_columns.hpp>
#include <neo/fft/convolution/overlap_add.hpp>
#include <neo/fft/convolution/overlap_save.hpp>
#include <neo/math/complex.hpp>
#include <neo/math/next_power_of_two.hpp>

#include <concepts>

namespace neo::fft {

template<std::floating_point Float, typename Overlap>
struct sparse_uniform_partitioned_convolver
{
    using value_type   = Float;
    using overlap_type = Overlap;

    using real_type    = Float;
    using complex_type = std::complex<Float>;
    using size_type    = std::size_t;

    sparse_uniform_partitioned_convolver() = default;

    template<std::predicate<std::size_t, std::size_t, complex_type> Sparsity>
    auto filter(in_matrix auto filter, Sparsity sparsity) -> void;
    auto operator()(inout_vector auto block) -> void;

private:
    sparse_matrix<complex_type> _filter;
    stdex::mdarray<complex_type, stdex::dextents<size_type, 1>> _accumulator;
    stdex::mdarray<complex_type, stdex::dextents<size_type, 2>> _fdl;

    Overlap _overlap{1, 1};
};

template<std::floating_point Float, typename Overlap>
template<std::predicate<std::size_t, std::size_t, std::complex<Float>> Sparsity>
auto sparse_uniform_partitioned_convolver<Float, Overlap>::filter(in_matrix auto filter, Sparsity sparsity) -> void
{
    _filter      = sparse_matrix<complex_type>{filter, sparsity};
    _overlap     = Overlap{filter.extent(1) - 1, filter.extent(1) - 1};
    _fdl         = stdex::mdarray<complex_type, stdex::dextents<size_type, 2>>{filter.extents()};
    _accumulator = stdex::mdarray<complex_type, stdex::dextents<size_type, 1>>{filter.extent(1)};
}

template<std::floating_point Float, typename Overlap>
auto sparse_uniform_partitioned_convolver<Float, Overlap>::operator()(inout_vector auto block) -> void
{
    _overlap(block, [this](inout_vector auto inout) {
        auto const fdl         = _fdl.to_mdspan();
        auto const accumulator = _accumulator.to_mdspan();

        shift_rows_up(fdl);
        copy(inout, stdex::submdspan(fdl, 0, stdex::full_extent));
        multiply_elements_add_columns(fdl, _filter, accumulator);
        copy(accumulator, inout);
    });
}

template<std::floating_point Float>
using sparse_upols_convolver = sparse_uniform_partitioned_convolver<Float, overlap_save<Float>>;

template<std::floating_point Float>
using sparse_upola_convolver = sparse_uniform_partitioned_convolver<Float, overlap_add<Float>>;

}  // namespace neo::fft
