#include "upols_convolver.hpp"

#include <neo/fft/algorithm/copy.hpp>
#include <neo/fft/algorithm/fill.hpp>
#include <neo/fft/algorithm/shift_rows_left.hpp>
#include <neo/fft/math/next_power_of_two.hpp>

#include <functional>

namespace neo::fft {

static auto multiply_and_accumulate_row(
    Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<std::size_t, 1>> lhs,
    Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<std::size_t, 1>> rhs,
    Kokkos::mdspan<std::complex<float>, Kokkos::dextents<std::size_t, 1>> accumulator
)
{
    auto* NEO_FFT_RESTRICT acc         = accumulator.data_handle();
    auto const* NEO_FFT_RESTRICT left  = lhs.data_handle();
    auto const* NEO_FFT_RESTRICT right = rhs.data_handle();
    for (decltype(lhs.size()) i{0}; i < lhs.size(); ++i) { acc[i] += left[i] * right[i]; }
}

static auto multiply_and_accumulate(
    Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<std::size_t, 2>> lhs,
    Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<std::size_t, 2>> rhs,
    Kokkos::mdspan<std::complex<float>, Kokkos::dextents<std::size_t, 1>> accumulator,
    std::size_t shift
)
{
    assert(lhs.extents() == rhs.extents());
    assert(lhs.extent(1) > 0);
    assert(shift < lhs.extent(0));

    for (auto row{0U}; row <= shift; ++row) {
        multiply_and_accumulate_row(
            KokkosEx::submdspan(lhs, row, Kokkos::full_extent),
            KokkosEx::submdspan(rhs, shift - row, Kokkos::full_extent),
            accumulator
        );
    }

    for (auto row{shift + 1}; row < lhs.extent(0); ++row) {
        auto const offset    = row - shift;
        auto const offsetRow = lhs.extent(0) - offset;
        multiply_and_accumulate_row(
            KokkosEx::submdspan(lhs, row, Kokkos::full_extent),
            KokkosEx::submdspan(rhs, offsetRow, Kokkos::full_extent),
            accumulator
        );
    }
}

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
    auto inout     = Kokkos::mdspan<float, Kokkos::dextents<std::size_t, 1>>{block.data(), block.size()};
    auto leftHalf  = Kokkos::mdspan<float, Kokkos::dextents<std::size_t, 1>>{_window.data(), block.size()};
    auto rightHalf = Kokkos::mdspan<float, Kokkos::dextents<std::size_t, 1>>{_window.data() + blockSize, block.size()};
    copy(rightHalf, leftHalf);
    copy(inout, rightHalf);

    // 2B-point R2C-FFT
    std::invoke(*_rfft, _window, _rfftBuf);

    // Copy to FDL
    for (auto i{0U}; i < _fdl.extent(1); ++i) { _fdl(_fdlIndex, i) = _rfftBuf[i] / float(_rfft->size()); }

    // DFT-spectrum additions
    fill(_accumulator.to_mdspan(), 0.0F);
    multiply_and_accumulate(_fdl.to_mdspan(), _filter, _accumulator.to_mdspan(), _fdlIndex);

    // All contents (DFT spectra) in the FDL are shifted up by one slot.
    ++_fdlIndex;
    if (_fdlIndex == _fdl.extent(0)) { _fdlIndex = 0; }

    // 2B-point C2R-IFFT
    std::invoke(*_rfft, std::span{_accumulator.data(), _accumulator.size()}, _irfftBuf);

    // Copy blockSize samples to output
    auto reconstructed = Kokkos::mdspan<float, Kokkos::dextents<std::size_t, 1>>{
        std::prev(std::next(_irfftBuf.data(), static_cast<std::ptrdiff_t>(_irfftBuf.size())), blockSize),
        block.size(),
    };
    copy(reconstructed, inout);
}

}  // namespace neo::fft
