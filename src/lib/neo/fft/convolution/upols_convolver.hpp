#pragma once

#include <neo/fft/container/mdspan.hpp>
#include <neo/fft/container/sparse_matrix.hpp>
#include <neo/fft/convolution/overlap_save.hpp>
#include <neo/fft/math/ilog2.hpp>
#include <neo/fft/transform.hpp>

#include <algorithm>
#include <complex>
#include <functional>
#include <memory>
#include <vector>

namespace neo::fft {

struct upols_convolver
{
    upols_convolver() = default;

    auto filter(KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> filter) -> void;
    auto operator()(std::span<float> block) -> void;

private:
    std::size_t _fdlIndex{0};
    KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>> _fdl;
    KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> _filter;
    KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 1>> _accumulator;

    overlap_save<float> _overlapSave;
};

}  // namespace neo::fft
