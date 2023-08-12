#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/algorithm/fill.hpp>
#include <neo/algorithm/multiply_sum_columns.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/container/sparse_matrix.hpp>
#include <neo/fft/convolution/overlap_add.hpp>
#include <neo/fft/convolution/overlap_save.hpp>
#include <neo/math/complex.hpp>

namespace neo::fft {

template<std::floating_point Float, typename Overlap = overlap_save<Float>>
struct uniform_partitioned_convolver
{
    uniform_partitioned_convolver() = default;

    auto filter(in_matrix auto filter) -> void;
    auto operator()(in_vector auto block) -> void;

private:
    Overlap _overlap{1, 1};

    KokkosEx::mdspan<std::complex<Float> const, Kokkos::dextents<size_t, 2>> _filter;
    KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>> _accumulator;
    KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 2>> _fdl;
    std::size_t _fdlIndex{0};
};

template<std::floating_point Float, typename Overlap>
auto uniform_partitioned_convolver<Float, Overlap>::filter(in_matrix auto filter) -> void
{
    _overlap     = Overlap{filter.extent(1) - 1, filter.extent(1) - 1};
    _fdl         = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 2>>{filter.extents()};
    _accumulator = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>>{filter.extent(1)};
    _filter      = filter;
    _fdlIndex    = 0;
}

template<std::floating_point Float, typename Overlap>
auto uniform_partitioned_convolver<Float, Overlap>::operator()(in_vector auto block) -> void
{
    _overlap(block, [this](inout_vector auto inout) {
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

template<std::floating_point Float>
using upols_convolver = uniform_partitioned_convolver<Float, overlap_save<Float>>;

template<std::floating_point Float>
using upola_convolver = uniform_partitioned_convolver<Float, overlap_add<Float>>;

}  // namespace neo::fft
