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

    template<std::regular_invocable<std::size_t, std::size_t, complex_type> SparsityFilter>
    auto filter(in_matrix auto filter, SparsityFilter sparsity) -> void;
    auto operator()(inout_vector auto block) -> void;

private:
    sparse_matrix<complex_type> _filter;
    KokkosEx::mdarray<complex_type, Kokkos::dextents<size_type, 1>> _accumulator;
    KokkosEx::mdarray<complex_type, Kokkos::dextents<size_type, 2>> _fdl;
    std::size_t _fdlIndex{0};

    overlap_save<Float> _overlapSave{1, 1};
};

template<std::floating_point Float>
template<std::regular_invocable<std::size_t, std::size_t, std::complex<Float>> SparsityFilter>
auto sparse_upols_convolver<Float>::filter(in_matrix auto filter, SparsityFilter sparsity) -> void
{
    _fdlIndex    = 0;
    _overlapSave = overlap_save<Float>{filter.extent(1) - 1, filter.extent(1) - 1};
    _filter      = sparse_matrix<complex_type>{filter, sparsity};
    _fdl         = KokkosEx::mdarray<complex_type, Kokkos::dextents<size_type, 2>>{filter.extents()};
    _accumulator = KokkosEx::mdarray<complex_type, Kokkos::dextents<size_type, 1>>{filter.extent(1)};
}

template<std::floating_point Float>
auto sparse_upols_convolver<Float>::operator()(inout_vector auto block) -> void
{
    _overlapSave(block, [this](inout_vector auto inout) {
        auto const fdl         = _fdl.to_mdspan();
        auto const accumulator = _accumulator.to_mdspan();

        copy(inout, KokkosEx::submdspan(fdl, _fdlIndex, Kokkos::full_extent));
        fill(accumulator, Float(0));
        multiply_sum_columns(fdl, _filter, accumulator, _fdlIndex);
        copy(accumulator, inout);

        ++_fdlIndex;
        if (_fdlIndex == fdl.extent(0)) {
            _fdlIndex = 0;
        }
    });
}

}  // namespace neo::fft
