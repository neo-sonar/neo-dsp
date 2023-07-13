#include "upols_convolver.hpp"

namespace neo::fft
{

static auto multiply_and_accumulate_row(std::span<std::complex<float> const> lhs,
                                        std::span<std::complex<float> const> rhs,
                                        std::span<std::complex<float>> accumulator)
{
    auto* NEO_FFT_RESTRICT acc         = accumulator.data();
    auto const* NEO_FFT_RESTRICT left  = lhs.data();
    auto const* NEO_FFT_RESTRICT right = rhs.data();
    for (decltype(lhs.size()) i{0}; i < lhs.size(); ++i) { acc[i] += left[i] * right[i]; }
}

static auto multiply_and_accumulate(Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<std::size_t, 2>> lhs,
                                    Kokkos::mdspan<std::complex<float> const, Kokkos::dextents<std::size_t, 2>> rhs,
                                    std::span<std::complex<float>> accumulator, std::size_t shift)
{
    assert(lhs.extents() == rhs.extents());
    assert(lhs.extent(1) > 0);
    assert(shift < lhs.extent(0));

    auto getRow = [](auto const& matrix, size_t row) {
        return std::span<std::complex<float> const>{std::addressof(matrix(row, 0)), matrix.extent(1)};
    };

    // First loop, so we don't need to clear the accumulator from previous iteration
    multiply_and_accumulate_row(getRow(lhs, 0), getRow(rhs, shift), accumulator);

    for (auto row{1U}; row <= shift; ++row)
    {
        multiply_and_accumulate_row(getRow(lhs, row), getRow(rhs, shift - row), accumulator);
    }

    for (auto row{shift + 1}; row < lhs.extent(0); ++row)
    {
        auto const offset    = row - shift;
        auto const offsetRow = lhs.extent(0) - offset - 1;
        multiply_and_accumulate_row(getRow(lhs, row), getRow(rhs, offsetRow), accumulator);
    }
}

auto upols_convolver::filter(KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> filter) -> void
{
    auto const K = std::bit_ceil((filter.extent(1) - 1U) * 2U);

    _fdl    = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>{filter.extents()};
    _filter = filter;

    _rfft = std::make_unique<rfft_radix2_plan<float>>(ilog2(K));
    _window.resize(K);
    _rfftBuf.resize(_window.size());
    _irfftBuf.resize(_window.size());
    _accumulator.resize(_filter.extent(1));

    _fdlIndex = 0;
}

auto upols_convolver::operator()(std::span<float> block) -> void
{
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
    multiply_and_accumulate(_fdl, _filter, _accumulator, _fdlIndex);

    // All contents (DFT spectra) in the FDL are shifted up by one slot.
    ++_fdlIndex;
    if (_fdlIndex == _fdl.extent(0)) { _fdlIndex = 0; }

    // 2B-point C2R-IFFT
    std::invoke(*_rfft, _accumulator, _irfftBuf);

    // Copy blockSize samples to output
    std::copy(std::prev(_irfftBuf.end(), blockSize), _irfftBuf.end(), block.begin());
}

}  // namespace neo::fft
