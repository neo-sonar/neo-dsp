#include "upols_convolver.hpp"

#include <neo/fft/algorithm/copy.hpp>
#include <neo/fft/algorithm/fill.hpp>
#include <neo/fft/algorithm/multiply_elementwise_sum_columnwise.hpp>

namespace neo::fft {

auto upols_convolver::filter(KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> filter) -> void
{
    _overlapSave = overlap_save<float>{filter.extent(1) - 1};
    _fdl         = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>>{filter.extents()};
    _accumulator = KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 1>>{filter.extent(1)};
    _filter      = filter;
    _fdlIndex    = 0;
}

auto upols_convolver::operator()(std::span<float> block) -> void
{
    _overlapSave(block, [this](auto inout) {
        auto const fdl         = _fdl.to_mdspan();
        auto const accumulator = _accumulator.to_mdspan();

        // Copy to FDL
        // copy(inout, KokkosEx::submdspan(_fdl.to_mdspan(), _fdlIndex, Kokkos::full_extent));
        for (auto i{0U}; i < fdl.extent(1); ++i) { fdl(_fdlIndex, i) = inout[i]; }

        // DFT-spectrum additions
        fill(accumulator, 0.0F);
        multiply_elementwise_sum_columnwise(fdl, _filter, accumulator, _fdlIndex);

        // Copy to output
        copy(accumulator, inout);

        // All contents (DFT spectra) in the FDL are shifted up by one slot.
        ++_fdlIndex;
        if (_fdlIndex == fdl.extent(0)) { _fdlIndex = 0; }
    });
}

}  // namespace neo::fft
