#include "sparse_upols_convolver.hpp"

#include <neo/fft/algorithm/multiply_elementwise_sum_columnwise.hpp>
#include <neo/fft/math/next_power_of_two.hpp>

#include <functional>
#include <random>

namespace neo::fft {

auto sparse_upols_convolver::filter(
    KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> filter,
    std::function<bool(std::size_t, std::size_t, std::complex<float>)> const& sparsiyFilter
) -> void
{
    auto const K = next_power_of_two((filter.extent(1) - 1U) * 2U);

    _fdl    = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>{filter.extents()};
    _filter = sparse_matrix<std::complex<float>>{filter, sparsiyFilter};

    _rfft = std::make_unique<rfft_radix2_plan<float>>(ilog2(K));
    _window.resize(K);
    _rfftBuf.resize(_window.size());
    _irfftBuf.resize(_window.size());
    _accumulator.resize(_filter.columns());

    _fdlIndex = 0;
}

auto sparse_upols_convolver::operator()(std::span<float> block) -> void
{
    assert(_fdlIndex < _fdl.extent(0));
    assert(_fdl.extent(1) > 0);
    assert(_fdl.extent(0) == _filter.rows());
    assert(_fdl.extent(1) == _filter.columns());
    assert(block.size() * 2U == _window.size());

    auto const blockSize = std::ssize(block);

    // Time domain input buffer
    std::shift_left(_window.begin(), _window.end(), blockSize);
    std::copy(block.begin(), block.end(), std::prev(_window.end(), blockSize));

    // 2B-point R2C-FFT
    std::invoke(*_rfft, _window, _rfftBuf);

    // Copy to FDL
    for (auto i{0U}; i < _fdl.extent(1); ++i) { _fdl(_fdlIndex, i) = _rfftBuf[i] / float(_rfft->size()); }

    // DFT-spectrum additions
    std::fill(_accumulator.begin(), _accumulator.end(), 0.0F);
    multiply_elementwise_sum_columnwise(_fdl.to_mdspan(), _filter, std::span{_accumulator}, _fdlIndex);

    // All contents (DFT spectra) in the FDL are shifted up by one slot.
    ++_fdlIndex;
    if (_fdlIndex == _fdl.extent(0)) { _fdlIndex = 0; }

    // 2B-point C2R-IFFT
    std::invoke(*_rfft, _accumulator, _irfftBuf);

    // Copy blockSize samples to output
    std::copy(std::prev(_irfftBuf.end(), blockSize), _irfftBuf.end(), block.begin());
}

}  // namespace neo::fft
