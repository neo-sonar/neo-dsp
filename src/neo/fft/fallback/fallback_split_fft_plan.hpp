// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/fft.hpp>

namespace neo::fft {

template<std::floating_point Float>
struct fallback_split_fft_plan
{
    using value_type = Float;
    using size_type  = std::size_t;

    explicit fallback_split_fft_plan(size_type order) : _order{order} {}

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return size_type(1) << order(); }

    template<inout_vector_of<Float> InOutVec>
    auto operator()(split_complex<InOutVec> x, direction dir) noexcept -> void
    {
        assert(std::cmp_equal(x.real.extent(0), size()));
        assert(neo::detail::extents_equal(x.real, x.imag));

        auto const buf = _buffer.to_mdspan();
        for (auto i{0}; i < static_cast<int>(x.real.extent(0)); ++i) {
            buf[i] = std::complex<Float>{x.real[i], x.imag[i]};
        }

        _fft(buf, dir);

        for (auto i{0}; i < static_cast<int>(x.real.extent(0)); ++i) {
            x.real[i] = buf[i].real();
            x.imag[i] = buf[i].imag();
        }
    }

    template<in_vector_of<Float> InVec, out_vector_of<Float> OutVec>
    auto operator()(split_complex<InVec> in, split_complex<OutVec> out, direction dir) noexcept -> void
    {
        assert(std::cmp_equal(in.real.extent(0), size()));
        assert(neo::detail::extents_equal(in.real, in.imag, out.real, out.imag));

        copy(in.real, out.real);
        copy(in.imag, out.imag);
        (*this)(out, dir);
    }

private:
    size_type _order;
    fft_plan<std::complex<Float>> _fft{_order};
    stdex::mdarray<std::complex<Float>, stdex::dextents<size_t, 1>> _buffer{size()};
};

}  // namespace neo::fft
