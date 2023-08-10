#include "upols_convolver.hpp"

#include <neo/fft/algorithm/copy.hpp>
#include <neo/fft/algorithm/fill.hpp>
#include <neo/fft/algorithm/multiply_elementwise_sum_columnwise.hpp>
#include <neo/fft/math/next_power_of_two.hpp>

#include <functional>

namespace neo::fft {

auto upols_convolver::filter(KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> filter) -> void
{
    auto const K = next_power_of_two((filter.extent(1) - 1U) * 2U);

    _fdl         = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>{filter.extents()};
    _accumulator = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 1>>{filter.extent(1)};
    _filter      = filter;

    _rfft = std::make_unique<rfft_radix2_plan<float>>(ilog2(K));
    _window.resize(K);
    _rfftBuf.resize(_window.size());
    _irfftBuf.resize(_window.size());

    _fdlIndex = 0;
}

auto upols_convolver::operator()(std::span<float> block) -> void
{
    assert(block.size() * 2U == _window.size());

    auto const blockSize = std::ssize(block);

    // Time domain input buffer
    auto inout     = Kokkos::mdspan{block.data(), Kokkos::extents{block.size()}};
    auto leftHalf  = Kokkos::mdspan{_window.data(), Kokkos::extents{block.size()}};
    auto rightHalf = Kokkos::mdspan{_window.data() + blockSize, Kokkos::extents{block.size()}};
    copy(rightHalf, leftHalf);
    copy(inout, rightHalf);

    // 2B-point R2C-FFT
    std::invoke(*_rfft, _window, _rfftBuf);

    // Copy to FDL
    for (auto i{0U}; i < _fdl.extent(1); ++i) { _fdl(_fdlIndex, i) = _rfftBuf[i] / float(_rfft->size()); }

    // DFT-spectrum additions
    fill(_accumulator.to_mdspan(), 0.0F);
    multiply_elementwise_sum_columnwise(_fdl.to_mdspan(), _filter, _accumulator.to_mdspan(), _fdlIndex);

    // All contents (DFT spectra) in the FDL are shifted up by one slot.
    ++_fdlIndex;
    if (_fdlIndex == _fdl.extent(0)) { _fdlIndex = 0; }

    // 2B-point C2R-IFFT
    std::invoke(*_rfft, std::span{_accumulator.data(), _accumulator.size()}, _irfftBuf);

    // Copy blockSize samples to output
    auto reconstructed = Kokkos::mdspan{
        std::prev(std::next(_irfftBuf.data(), static_cast<std::ptrdiff_t>(_irfftBuf.size())), blockSize),
        Kokkos::extents{block.size()},
    };
    copy(reconstructed, inout);
}

}  // namespace neo::fft
