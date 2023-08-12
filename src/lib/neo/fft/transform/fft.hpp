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

template<typename Complex, typename Kernel = radix2_kernel_v3>
struct fft_radix2_plan
{
    using complex_type = Complex;
    using size_type    = std::size_t;

    explicit fft_radix2_plan(size_type order) : _order{order} {}

    [[nodiscard]] auto size() const noexcept -> size_type { return _size; }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    template<inout_vector InOutVec>
        requires(std::same_as<typename InOutVec::value_type, Complex>)
    auto operator()(InOutVec vec, direction dir) -> void
    {
        NEO_FFT_PRECONDITION(std::cmp_equal(vec.size(), _size));

        bit_reverse_permutation(vec, _index_table);

        if (dir == direction::forward) {
            Kernel{}(vec, _twiddles.to_mdspan());
        } else {
            Kernel{}(vec, conjugate_view{_twiddles.to_mdspan()});
        }
    }

    template<in_vector InVec, out_vector OutVec>
        requires(std::same_as<typename InVec::value_type, Complex> and std::same_as<typename OutVec::value_type, Complex>)
    auto operator()(InVec in, OutVec out, direction dir) -> void
    {
        NEO_FFT_PRECONDITION(std::cmp_equal(in.size(), _size));
        NEO_FFT_PRECONDITION(std::cmp_equal(out.size(), _size));

        copy(in, out);
        (*this)(out, dir);
    }

private:
    size_type _order;
    size_type _size{1ULL << _order};
    std::vector<size_type> _index_table{make_bit_reversed_index_table(_size)};
    KokkosEx::mdarray<Complex, Kokkos::dextents<std::size_t, 1>> _twiddles{make_radix2_twiddles<Complex>(_size)};
};

inline constexpr auto fft_radix2 = [](auto const& kernel, inout_vector auto x, auto const& twiddles) -> void {
    bit_reverse_permutation(x);
    kernel(x, twiddles);
};

}  // namespace neo::fft
