#pragma once

#include <neo/fft/algorithm/copy.hpp>
#include <neo/fft/algorithm/fill.hpp>
#include <neo/fft/container/mdspan.hpp>
#include <neo/fft/math/ilog2.hpp>
#include <neo/fft/math/next_power_of_two.hpp>
#include <neo/fft/transform.hpp>

#include <complex>
#include <functional>
#include <memory>
#include <vector>

namespace neo::fft {

template<std::floating_point Float>
struct overlap_save
{
    using callback_type = std::function<void(KokkosEx::mdspan<std::complex<Float>, Kokkos::dextents<size_t, 1>>)>;
    overlap_save()      = default;
    explicit overlap_save(std::size_t block_size);

    auto operator()(std::span<Float> block, callback_type const& callback) -> void;

private:
    std::size_t _blockSize{0};
    std::size_t _windowSize{_blockSize * 2UL};

    KokkosEx::mdarray<Float, Kokkos::dextents<size_t, 1>> _window{_windowSize};
    KokkosEx::mdarray<Float, Kokkos::dextents<size_t, 1>> _realBuffer{_windowSize};
    KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>> _complexBuffer{_windowSize};
    std::unique_ptr<rfft_radix2_plan<Float>> _rfft{std::make_unique<rfft_radix2_plan<Float>>(ilog2(_windowSize))};
};

template<std::floating_point Float>
overlap_save<Float>::overlap_save(std::size_t block_size) : _blockSize{block_size}
{}

template<std::floating_point Float>
auto overlap_save<Float>::operator()(std::span<Float> block, callback_type const& callback) -> void
{
    assert(block.size() * 2U == _window.size());

    auto const inout  = Kokkos::mdspan{block.data(), Kokkos::extents{block.size()}};
    auto const window = _window.to_mdspan();

    // Time domain input buffer
    auto leftHalf  = KokkosEx::submdspan(window, std::tuple{0, _window.extent(0) / 2});
    auto rightHalf = KokkosEx::submdspan(window, std::tuple{_window.extent(0) / 2, _window.extent(0)});
    copy(rightHalf, leftHalf);
    copy(inout, rightHalf);

    // 2B-point R2C-FFT
    auto const complexBuffer = _complexBuffer.to_mdspan();
    auto const realBuffer    = _realBuffer.to_mdspan();
    std::invoke(*_rfft, window, complexBuffer);

    // Copy to FDL
    auto const scale     = 1.0F / static_cast<Float>(_rfft->size());
    auto const numCoeffs = _rfft->size() / 2 + 1;
    for (auto i{0U}; i < numCoeffs; ++i) { complexBuffer[i] *= scale; }

    if (callback) { callback(KokkosEx::submdspan(complexBuffer, std::tuple{0, numCoeffs})); }

    // 2B-point C2R-IFFT
    std::invoke(*_rfft, complexBuffer, realBuffer);

    // Copy blockSize samples to output
    auto out = KokkosEx::submdspan(realBuffer, std::tuple{realBuffer.extent(0) - block.size(), realBuffer.extent(0)});
    copy(out, inout);
}

}  // namespace neo::fft
