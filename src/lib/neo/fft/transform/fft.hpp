#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/transform/conjugate_view.hpp>
#include <neo/fft/transform/direction.hpp>
#include <neo/fft/transform/kernel/radix2.hpp>
#include <neo/fft/transform/reorder.hpp>
#include <neo/fft/transform/twiddle.hpp>
#include <neo/math/complex.hpp>

#include <utility>

namespace neo::fft {

template<typename Plan, inout_vector InOutVec>
constexpr auto fft(Plan& plan, InOutVec inout) -> void
{
    NEO_EXPECTS(std::cmp_equal(inout.size(), plan.size()));
    plan(inout, direction::forward);
}

template<typename Plan, in_vector InVec, out_vector OutVec>
constexpr auto fft(Plan& plan, InVec input, OutVec output) -> void
{
    NEO_EXPECTS(std::cmp_equal(input.size(), plan.size()));
    NEO_EXPECTS(std::cmp_equal(output.size(), plan.size()));
    copy(input, output);
    fft(plan, output);
}

template<typename Plan, inout_vector InOutVec>
constexpr auto ifft(Plan& plan, InOutVec inout) -> void
{
    NEO_EXPECTS(std::cmp_equal(inout.size(), plan.size()));
    plan(inout, direction::backward);
}

template<typename Plan, in_vector InVec, out_vector OutVec>
constexpr auto ifft(Plan& plan, InVec input, OutVec output) -> void
{
    NEO_EXPECTS(std::cmp_equal(input.size(), plan.size()));
    NEO_EXPECTS(std::cmp_equal(output.size(), plan.size()));
    copy(input, output);
    ifft(plan, output);
}

template<typename Complex, typename Kernel = radix2_kernel_v3>
struct fft_radix2_plan
{
    using complex_type = Complex;
    using size_type    = std::size_t;

    explicit fft_radix2_plan(size_type order, direction default_direction = direction::forward);

    [[nodiscard]] auto order() const noexcept -> size_type;
    [[nodiscard]] auto size() const noexcept -> size_type;

    template<inout_vector InOutVec>
        requires std::same_as<typename InOutVec::value_type, Complex>
    auto operator()(InOutVec vec, direction dir) -> void;

private:
    size_type _order;
    size_type _size{1ULL << _order};
    direction _default_direction;
    std::vector<size_type> _index_table{make_bit_reversed_index_table(_size)};
    KokkosEx::mdarray<Complex, Kokkos::dextents<size_type, 1>> _twiddles{
        make_radix2_twiddles<Complex>(_size, _default_direction),
    };
};

template<typename Complex, typename Kernel>
fft_radix2_plan<Complex, Kernel>::fft_radix2_plan(size_type order, direction default_direction)
    : _order{order}
    , _default_direction{default_direction}
{}

template<typename Complex, typename Kernel>
auto fft_radix2_plan<Complex, Kernel>::size() const noexcept -> size_type
{
    return _size;
}

template<typename Complex, typename Kernel>
auto fft_radix2_plan<Complex, Kernel>::order() const noexcept -> size_type
{
    return _order;
}

template<typename Complex, typename Kernel>
template<inout_vector InOutVec>
    requires std::same_as<typename InOutVec::value_type, Complex>
auto fft_radix2_plan<Complex, Kernel>::operator()(InOutVec vec, direction dir) -> void
{
    NEO_EXPECTS(std::cmp_equal(vec.size(), _size));

    bit_reverse_permutation(vec, _index_table);

    if (auto const kernel = Kernel{}; dir == _default_direction) {
        kernel(vec, _twiddles.to_mdspan());
    } else {
        kernel(vec, conjugate_view{_twiddles.to_mdspan()});
    }
}

inline constexpr auto execute_radix2_kernel = [](auto kernel, inout_vector auto x, auto const& twiddles) -> void {
    bit_reverse_permutation(x);
    kernel(x, twiddles);
};

}  // namespace neo::fft
