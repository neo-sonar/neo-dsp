#pragma once

#include <neo/fft/algorithm/copy.hpp>
#include <neo/fft/algorithm/fill.hpp>
#include <neo/fft/algorithm/multiply_elementwise_sum_columnwise.hpp>
#include <neo/fft/container/mdspan.hpp>
#include <neo/fft/container/sparse_matrix.hpp>
#include <neo/fft/convolution/overlap_save.hpp>

#include <complex>
#include <functional>
#include <memory>

namespace neo::fft {

template<std::floating_point Float>
struct upols_convolver
{
    upols_convolver() = default;

    auto filter(in_matrix auto filter) -> void;
    auto operator()(in_vector auto block) -> void;

private:
    KokkosEx::mdspan<std::complex<Float> const, Kokkos::dextents<size_t, 2>> _filter;
    KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>> _accumulator;
    KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 2>> _fdl;
    std::size_t _fdlIndex{0};

    overlap_save<Float> _overlapSave;
};

template<std::floating_point Float>
auto upols_convolver<Float>::filter(in_matrix auto filter) -> void
{
    _overlapSave = overlap_save<Float>{filter.extent(1) - 1};
    _fdl         = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 2>>{filter.extents()};
    _accumulator = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>>{filter.extent(1)};
    _filter      = filter;
    _fdlIndex    = 0;
}

template<std::floating_point Float>
auto upols_convolver<Float>::operator()(in_vector auto block) -> void
{
    _overlapSave(block, [this](inout_vector auto inout) {
        auto const fdl         = _fdl.to_mdspan();
        auto const accumulator = _accumulator.to_mdspan();

        copy(inout, KokkosEx::submdspan(fdl, _fdlIndex, Kokkos::full_extent));
        fill(accumulator, Float(0));
        multiply_elementwise_sum_columnwise(fdl, _filter, accumulator, _fdlIndex);
        copy(accumulator, inout);

        ++_fdlIndex;
        if (_fdlIndex == fdl.extent(0)) {
            _fdlIndex = 0;
        }
    });
}

}  // namespace neo::fft
