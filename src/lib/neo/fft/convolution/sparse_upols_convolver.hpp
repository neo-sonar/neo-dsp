#pragma once

#include <neo/fft/algorithm/multiply_elementwise_sum_columnwise.hpp>
#include <neo/fft/container/mdspan.hpp>
#include <neo/fft/container/sparse_matrix.hpp>
#include <neo/fft/convolution/overlap_save.hpp>
#include <neo/fft/math/next_power_of_two.hpp>

#include <complex>
#include <functional>
#include <memory>

namespace neo::fft {

struct sparse_upols_convolver
{
    sparse_upols_convolver() = default;

    auto filter(
        KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> filter,
        std::function<bool(std::size_t, std::size_t, std::complex<float>)> const& sparsityFilter
    ) -> void;
    auto operator()(std::span<float> block) -> void;

private:
    sparse_matrix<std::complex<float>> _filter;
    KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 1>> _accumulator;
    KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>> _fdl;
    std::size_t _fdlIndex{0};

    overlap_save<float> _overlapSave;
};

inline auto sparse_upols_convolver::filter(
    KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> filter,
    std::function<bool(std::size_t, std::size_t, std::complex<float>)> const& sparsityFilter
) -> void
{
    _fdlIndex    = 0;
    _overlapSave = overlap_save<float>{filter.extent(1) - 1};
    _filter      = sparse_matrix<std::complex<float>>{filter, sparsityFilter};
    _fdl         = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>{filter.extents()};
    _accumulator = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 1>>{filter.extent(1)};
}

inline auto sparse_upols_convolver::operator()(std::span<float> block) -> void
{
    _overlapSave(block, [this](inout_vector auto inout) {
        auto const fdl         = _fdl.to_mdspan();
        auto const accumulator = _accumulator.to_mdspan();
        auto const acc         = std::span{accumulator.data_handle(), accumulator.size()};

        copy(inout, KokkosEx::submdspan(fdl, _fdlIndex, Kokkos::full_extent));
        fill(accumulator, 0.0F);
        multiply_elementwise_sum_columnwise(fdl, _filter, acc, _fdlIndex);
        copy(accumulator, inout);

        ++_fdlIndex;
        if (_fdlIndex == fdl.extent(0)) { _fdlIndex = 0; }
    });
}

}  // namespace neo::fft
