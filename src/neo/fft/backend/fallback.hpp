#pragma once

#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/bitrevorder.hpp>
#include <neo/fft/conjugate_view.hpp>
#include <neo/fft/direction.hpp>
#include <neo/fft/kernel/radix2.hpp>
#include <neo/fft/twiddle.hpp>

#include <cassert>

namespace neo::fft {

template<typename Complex, typename Kernel = kernel::c2c_dit2_v3>
struct fallback_fft_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    explicit fallback_fft_plan(size_type order, direction default_direction = direction::forward);

    [[nodiscard]] auto order() const noexcept -> size_type;
    [[nodiscard]] auto size() const noexcept -> size_type;

    template<inout_vector Vec>
        requires std::same_as<typename Vec::value_type, Complex>
    auto operator()(Vec x, direction dir) noexcept -> void;

private:
    size_type _order;
    size_type _size{1ULL << _order};
    direction _default_direction;
    bitrevorder_plan _reorder{_order};
    stdex::mdarray<Complex, stdex::dextents<size_type, 1>> _twiddles{
        make_radix2_twiddles<Complex>(_size, _default_direction),
    };
};

template<typename Complex, typename Kernel>
fallback_fft_plan<Complex, Kernel>::fallback_fft_plan(size_type order, direction default_direction)
    : _order{order}
    , _default_direction{default_direction}
{}

template<typename Complex, typename Kernel>
auto fallback_fft_plan<Complex, Kernel>::size() const noexcept -> size_type
{
    return _size;
}

template<typename Complex, typename Kernel>
auto fallback_fft_plan<Complex, Kernel>::order() const noexcept -> size_type
{
    return _order;
}

template<typename Complex, typename Kernel>
template<inout_vector Vec>
    requires std::same_as<typename Vec::value_type, Complex>
auto fallback_fft_plan<Complex, Kernel>::operator()(Vec x, direction dir) noexcept -> void
{
    assert(std::cmp_equal(x.size(), _size));

    _reorder(x);

    if (auto const kernel = Kernel{}; dir == _default_direction) {
        kernel(x, _twiddles.to_mdspan());
    } else {
        kernel(x, conjugate_view{_twiddles.to_mdspan()});
    }
}

}  // namespace neo::fft
