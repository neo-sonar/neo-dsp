#pragma once

#include "neo/convolution/container/mdspan.hpp"
#include "neo/convolution/container/sparse_matrix.hpp"
#include "neo/fft.hpp"

#include <algorithm>
#include <memory>
#include <vector>

namespace neo::fft {

struct sparse_upols_convolver
{
    explicit sparse_upols_convolver(float thresholdDB) : _thresholdDB{thresholdDB} {}

    auto filter(KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> filter) -> void;
    auto operator()(std::span<float> block) -> void;

private:
    float _thresholdDB;

    std::vector<float> _window;
    std::vector<std::complex<float>> _accumulator;
    KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>> _fdl;
    sparse_matrix<std::complex<float>> _filter;
    std::size_t _fdlIndex{0};

    std::unique_ptr<rfft_radix2_plan<float>> _rfft;
    std::vector<std::complex<float>> _rfftBuf;
    std::vector<float> _irfftBuf;
};

}  // namespace neo::fft
