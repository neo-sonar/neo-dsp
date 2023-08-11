#pragma once

#include <neo/fft/config.hpp>

#include <neo/fft/algorithm/copy.hpp>
#include <neo/fft/algorithm/fill.hpp>
#include <neo/fft/algorithm/scale.hpp>
#include <neo/fft/container/mdspan.hpp>
#include <neo/fft/math/divide_round_up.hpp>
#include <neo/fft/math/ilog2.hpp>
#include <neo/fft/math/next_power_of_two.hpp>
#include <neo/fft/transform.hpp>

#include <complex>
#include <functional>

namespace neo::fft {

template<std::floating_point Float>
struct overlap_add
{
    using size_type = std::size_t;

    overlap_add() = default;
    overlap_add(size_type block_size, size_type filter_size);

    [[nodiscard]] auto block_size() const noexcept -> size_type;
    [[nodiscard]] auto filter_size() const noexcept -> size_type;
    [[nodiscard]] auto transform_size() const noexcept -> size_type;
    [[nodiscard]] auto overlaps() const noexcept -> size_type;

    auto operator()(
        inout_vector auto block,
        std::invocable<KokkosEx::mdspan<std::complex<Float>, Kokkos::dextents<size_t, 1>>> auto callback
    ) -> void;

private:
    size_type _block_size{0};
    size_type _filter_size{0};

    rfft_plan<Float> _rfft{ilog2(next_power_of_two(_block_size + _filter_size - 1UL))};
    KokkosEx::mdarray<Float, Kokkos::dextents<size_t, 1>> _real_buffer{_rfft.size()};
    KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>> _complex_buffer{_rfft.size()};

    size_type _overlapIdx{0};
    KokkosEx::mdarray<Float, Kokkos::dextents<size_t, 2>> _overlaps{
        divide_round_up(_block_size + _filter_size - 1UL, _block_size),
        _block_size + _filter_size - 1UL,
    };
};

template<std::floating_point Float>
overlap_add<Float>::overlap_add(size_type block_size, size_type filter_size)
    : _block_size{block_size}
    , _filter_size{filter_size}
{}

template<std::floating_point Float>
auto overlap_add<Float>::block_size() const noexcept -> size_type
{
    return _block_size;
}

template<std::floating_point Float>
auto overlap_add<Float>::filter_size() const noexcept -> size_type
{
    return _filter_size;
}

template<std::floating_point Float>
auto overlap_add<Float>::transform_size() const noexcept -> size_type
{
    return _rfft.size();
}

template<std::floating_point Float>
auto overlap_add<Float>::overlaps() const noexcept -> size_type
{
    return _overlaps.extent(0);
}

template<std::floating_point Float>
auto overlap_add<Float>::operator()(
    inout_vector auto block,
    std::invocable<KokkosEx::mdspan<std::complex<Float>, Kokkos::dextents<size_t, 1>>> auto callback
) -> void
{
    NEO_FFT_PRECONDITION(_rfft.size() == block.extent(0) * 2U);

    auto const real_buffer    = _real_buffer.to_mdspan();
    auto const complex_buffer = _complex_buffer.to_mdspan();

    fill(real_buffer, Float(0));
    copy(block, KokkosEx::submdspan(real_buffer, std::tuple{0, block.extent(1)}));

    _rfft(real_buffer, complex_buffer);

    auto const alpha  = 1.0F / static_cast<Float>(_rfft.size());
    auto const coeffs = KokkosEx::submdspan(complex_buffer, std::tuple{0, _rfft.size() / 2 + 1});
    scale(alpha, coeffs);

    callback(coeffs);

    _rfft(complex_buffer, real_buffer);

    copy(
        KokkosEx::submdspan(real_buffer, std::tuple{0, _overlaps.extent(1)}),
        KokkosEx::submdspan(_overlaps.to_mdspan(), _overlapIdx, Kokkos::full_extent)
    );
    copy(KokkosEx::submdspan(real_buffer, std::tuple{0, block.extent(0)}), block);

    ++_overlapIdx;
    if (_overlapIdx == _overlaps.extent(0)) {
        _overlapIdx = 0;
    }
}

}  // namespace neo::fft
