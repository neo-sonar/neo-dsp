// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>
#include <neo/fft/fallback/bitrevorder.hpp>
#include <neo/fft/fallback/conjugate_view.hpp>
#include <neo/fft/fallback/kernel/c2c_dit2.hpp>
#include <neo/fft/order.hpp>
#include <neo/fft/twiddle.hpp>
#include <neo/math/polar.hpp>

#include <cassert>
#include <numbers>

namespace neo::fft {

template<typename Complex, typename Kernel = kernel::c2c_dit2_v3>
struct fallback_fft_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    fallback_fft_plan(from_order_tag /*tag*/, size_type order);

    [[nodiscard]] static constexpr auto max_order() noexcept -> size_type;
    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type;

    [[nodiscard]] auto order() const noexcept -> size_type;
    [[nodiscard]] auto size() const noexcept -> size_type;

    template<inout_vector Vec>
        requires std::same_as<typename Vec::value_type, Complex>
    auto operator()(Vec x, direction dir) noexcept -> void;

private:
    [[nodiscard]] static auto check_order(size_type order) -> size_type;

    size_type _order;
    size_type _size{fft::size(order())};
    bitrevorder_plan _reorder{static_cast<size_t>(_order)};
    stdex::mdarray<Complex, stdex::dextents<size_type, 1>> _twiddles{
        make_twiddle_lut_radix2<Complex>(_size, direction::forward),
    };
};

template<typename Complex, typename Kernel>
fallback_fft_plan<Complex, Kernel>::fallback_fft_plan(from_order_tag /*tag*/, size_type order)
    : _order{check_order(order)}
{}

template<typename Complex, typename Kernel>
constexpr auto fallback_fft_plan<Complex, Kernel>::max_order() noexcept -> size_type
{
    return size_type{27};
}

template<typename Complex, typename Kernel>
constexpr auto fallback_fft_plan<Complex, Kernel>::max_size() noexcept -> size_type
{
    return fft::size(max_order());
}

template<typename Complex, typename Kernel>
auto fallback_fft_plan<Complex, Kernel>::order() const noexcept -> size_type
{
    return _order;
}

template<typename Complex, typename Kernel>
auto fallback_fft_plan<Complex, Kernel>::size() const noexcept -> size_type
{
    return _size;
}

template<typename Complex, typename Kernel>
template<inout_vector Vec>
    requires std::same_as<typename Vec::value_type, Complex>
auto fallback_fft_plan<Complex, Kernel>::operator()(Vec x, direction dir) noexcept -> void
{
    assert(std::cmp_equal(x.size(), _size));

    _reorder(x);

    if (auto const kernel = Kernel{}; dir == direction::forward) {
        kernel(x, _twiddles.to_mdspan());
    } else {
        kernel(x, conjugate_view{_twiddles.to_mdspan()});
    }
}

template<typename Complex, typename Kernel>
auto fallback_fft_plan<Complex, Kernel>::check_order(size_type order) -> size_type
{
    if (order > max_order()) {
        throw std::runtime_error{"fallback: unsupported order '" + std::to_string(int(order)) + "'"};
    }
    return order;
}

}  // namespace neo::fft
