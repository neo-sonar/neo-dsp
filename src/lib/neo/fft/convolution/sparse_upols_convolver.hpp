#pragma once

#include <neo/algorithm/multiply_sum_columns.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/container/sparse_matrix.hpp>
#include <neo/fft/convolution/overlap_save.hpp>
#include <neo/math/complex.hpp>
#include <neo/math/next_power_of_two.hpp>

#include <concepts>

namespace neo::fft {

template<std::floating_point Float>
struct sparse_upols_convolver
{
    using real_type    = Float;
    using complex_type = std::complex<Float>;
    using size_type    = std::size_t;

    sparse_upols_convolver() = default;

    template<std::predicate<std::size_t, std::size_t, complex_type> Sparsity>
    auto filter(in_matrix auto filter, Sparsity sparsity) -> void;
    auto operator()(inout_vector auto block) -> void;

private:
    sparse_matrix<complex_type> _filter;
    KokkosEx::mdarray<complex_type, Kokkos::dextents<size_type, 1>> _accumulator;
    KokkosEx::mdarray<complex_type, Kokkos::dextents<size_type, 2>> _fdl;

    overlap_save<Float> _overlap{1, 1};
};

template<std::floating_point Float>
template<std::predicate<std::size_t, std::size_t, std::complex<Float>> Sparsity>
auto sparse_upols_convolver<Float>::filter(in_matrix auto filter, Sparsity sparsity) -> void
{
    _filter      = sparse_matrix<complex_type>{filter, sparsity};
    _overlap     = overlap_save<Float>{filter.extent(1) - 1, filter.extent(1) - 1};
    _fdl         = KokkosEx::mdarray<complex_type, Kokkos::dextents<size_type, 2>>{filter.extents()};
    _accumulator = KokkosEx::mdarray<complex_type, Kokkos::dextents<size_type, 1>>{filter.extent(1)};
}

template<std::floating_point Float>
auto sparse_upols_convolver<Float>::operator()(inout_vector auto block) -> void
{
    _overlap(block, [this](inout_vector auto inout) {
        auto const fdl         = _fdl.to_mdspan();
        auto const accumulator = _accumulator.to_mdspan();

        shift_rows_up(fdl);
        copy(inout, KokkosEx::submdspan(fdl, 0, Kokkos::full_extent));
        fill(accumulator, Float(0));
        multiply_sum_columns(fdl, _filter, accumulator);
        copy(accumulator, inout);
    });
}

}  // namespace neo::fft
