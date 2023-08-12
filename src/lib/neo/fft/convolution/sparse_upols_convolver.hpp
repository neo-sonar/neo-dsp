#pragma once

#include <neo/fft/algorithm/multiply_sum_columns.hpp>
#include <neo/fft/container/mdspan.hpp>
#include <neo/fft/container/sparse_matrix.hpp>
#include <neo/fft/convolution/overlap_save.hpp>
#include <neo/fft/math/next_power_of_two.hpp>

#include <complex>
#include <concepts>

namespace neo::fft {

template<std::floating_point Float>
struct sparse_upols_convolver
{
    sparse_upols_convolver() = default;

    template<std::regular_invocable<std::size_t, std::size_t, std::complex<Float>> SparsityFilter>
    auto filter(in_matrix auto filter, SparsityFilter sparsity) -> void;
    auto operator()(inout_vector auto block) -> void;

private:
    sparse_matrix<std::complex<Float>> _filter;
    KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>> _accumulator;
    KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 2>> _fdl;
    std::size_t _fdlIndex{0};

    overlap_save<Float> _overlapSave{1, 1};
};

template<std::floating_point Float>
template<std::regular_invocable<std::size_t, std::size_t, std::complex<Float>> SparsityFilter>
auto sparse_upols_convolver<Float>::filter(in_matrix auto filter, SparsityFilter sparsity) -> void
{
    _fdlIndex    = 0;
    _overlapSave = overlap_save<Float>{filter.extent(1) - 1, filter.extent(1) - 1};
    _filter      = sparse_matrix<std::complex<Float>>{filter, sparsity};
    _fdl         = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 2>>{filter.extents()};
    _accumulator = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>>{filter.extent(1)};
}

template<std::floating_point Float>
auto sparse_upols_convolver<Float>::operator()(inout_vector auto block) -> void
{
    _overlapSave(block, [this](inout_vector auto inout) {
        auto const fdl         = _fdl.to_mdspan();
        auto const accumulator = _accumulator.to_mdspan();
        auto const acc         = std::span{accumulator.data_handle(), accumulator.size()};

        copy(inout, KokkosEx::submdspan(fdl, _fdlIndex, Kokkos::full_extent));
        fill(accumulator, Float(0));
        multiply_sum_columns(fdl, _filter, acc, _fdlIndex);
        copy(accumulator, inout);

        ++_fdlIndex;
        if (_fdlIndex == fdl.extent(0)) {
            _fdlIndex = 0;
        }
    });
}

}  // namespace neo::fft
