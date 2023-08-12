#pragma once

#include <neo/fft/config.hpp>

#include <neo/fft/algorithm/copy.hpp>
#include <neo/fft/algorithm/fill.hpp>
#include <neo/fft/algorithm/scale.hpp>
#include <neo/fft/container/mdspan.hpp>
#include <neo/fft/transform.hpp>
#include <neo/math/complex.hpp>
#include <neo/math/ilog2.hpp>
#include <neo/math/next_power_of_two.hpp>

#include <functional>

namespace neo::fft {

template<std::floating_point Float>
struct overlap_save
{
    using real_type    = Float;
    using complex_type = std::complex<Float>;
    using size_type    = std::size_t;

    overlap_save(size_type block_size, size_type filter_size);

    [[nodiscard]] auto block_size() const noexcept -> size_type;
    [[nodiscard]] auto filter_size() const noexcept -> size_type;
    [[nodiscard]] auto transform_size() const noexcept -> size_type;

    auto operator()(inout_vector auto block, auto callback) -> void;

private:
    size_type _block_size;
    size_type _filter_size;
    rfft_radix2_plan<Float> _rfft{ilog2(next_power_of_two(_block_size + _filter_size - 1UL))};

    KokkosEx::mdarray<Float, Kokkos::dextents<size_t, 1>> _window{_rfft.size()};
    KokkosEx::mdarray<Float, Kokkos::dextents<size_t, 1>> _real_buffer{_rfft.size()};
    KokkosEx::mdarray<complex_type, Kokkos::dextents<size_t, 1>> _complex_buffer{_rfft.size()};
};

template<std::floating_point Float>
overlap_save<Float>::overlap_save(size_type block_size, size_type filter_size)
    : _block_size{block_size}
    , _filter_size{filter_size}
{}

template<std::floating_point Float>
auto overlap_save<Float>::block_size() const noexcept -> size_type
{
    return _block_size;
}

template<std::floating_point Float>
auto overlap_save<Float>::filter_size() const noexcept -> size_type
{
    return _filter_size;
}

template<std::floating_point Float>
auto overlap_save<Float>::transform_size() const noexcept -> size_type
{
    return _rfft.size();
}

template<std::floating_point Float>
auto overlap_save<Float>::operator()(inout_vector auto block, auto callback) -> void
{
    NEO_FFT_PRECONDITION(block.extent(0) * 2U == _window.extent(0));

    // Time domain input buffer
    auto const window = _window.to_mdspan();
    auto left_half    = KokkosEx::submdspan(window, std::tuple{0, _window.extent(0) / 2});
    auto right_half   = KokkosEx::submdspan(window, std::tuple{_window.extent(0) / 2, _window.extent(0)});
    copy(right_half, left_half);
    copy(block, right_half);

    // 2B-point R2C-FFT
    auto const complex_buf = _complex_buffer.to_mdspan();
    _rfft(window, complex_buf);

    // Copy to FDL
    auto const num_coeffs = _rfft.size() / 2 + 1;
    auto const coeffs     = KokkosEx::submdspan(complex_buf, std::tuple{0, num_coeffs});
    scale(1.0F / static_cast<Float>(_rfft.size()), coeffs);

    // Apply processing
    callback(coeffs);

    // 2B-point C2R-IFFT
    auto const real_buf = _real_buffer.to_mdspan();
    _rfft(complex_buf, real_buf);

    // Copy blockSize samples to output
    auto out = KokkosEx::submdspan(real_buf, std::tuple{real_buf.extent(0) - block.size(), real_buf.extent(0)});
    copy(out, block);
}

}  // namespace neo::fft
