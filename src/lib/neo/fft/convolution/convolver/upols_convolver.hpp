#pragma once

#include "neo/fft/convolution/container/mdspan.hpp"
#include "neo/fft/convolution/container/sparse_matrix.hpp"
#include "neo/fft/transform.hpp"

#include <algorithm>
#include <complex>
#include <memory>
#include <vector>

namespace neo::fft {

struct upols_convolver
{
    upols_convolver() = default;

    auto filter(KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> filter) -> void;
    auto operator()(std::span<float> block) -> void;

private:
    std::vector<float> _window;

    std::size_t _fdlIndex{0};
    KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>> _fdl;
    KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> _filter;
    KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 1>> _accumulator;

    std::unique_ptr<rfft_radix2_plan<float>> _rfft;
    std::vector<std::complex<float>> _rfftBuf;
    std::vector<float> _irfftBuf;
};

struct stereo_upols_convolver
{
    stereo_upols_convolver() = default;

    auto filter(KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 3>> filter) -> void;
    auto operator()(KokkosEx::mdspan<float, Kokkos::dextents<size_t, 2>> block) -> void;

private:
    KokkosEx::mdarray<float, Kokkos::dextents<size_t, 2>> _window;
    KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>> _accumulator;
    KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 3>> _fdl;
    KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 3>> _filter;
    std::size_t _fdlIndex{0};

    std::unique_ptr<rfft_radix2_plan<float>> _fft;
    std::vector<std::complex<float>> _rfftBuf;
    std::vector<float> _irfftBuf;
};

}  // namespace neo::fft
