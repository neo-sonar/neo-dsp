// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/fft.hpp>
#include <neo/fft/order.hpp>

namespace neo::fft {

template<std::floating_point Float>
struct fallback_split_fft_plan
{
    using value_type = Float;
    using size_type  = std::size_t;

    explicit fallback_split_fft_plan(fft::order order) : _order{order} {}

    [[nodiscard]] auto order() const noexcept -> fft::order { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return fft::size(order()); }

    template<inout_vector_of<Float> InOutVec>
    auto operator()(split_complex<InOutVec> x, direction dir) noexcept -> void
    {
        assert(std::cmp_equal(x.real.extent(0), size()));
        assert(neo::detail::extents_equal(x.real, x.imag));

        _reorder(x);

        if (dir == direction::forward) {
            stage_0(x.real, x.imag);
        } else {
            stage_0(x.imag, x.real);
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
    auto stage_0(inout_vector_of<Float> auto xre, inout_vector_of<Float> auto xim) -> void
    {
        static constexpr auto const stage_length = 1;  // ipow<2>(0)
        static constexpr auto const stride       = 2;  // ipow<2>(0 + 1)

        for (auto k{0}; k < static_cast<int>(size()); k += stride) {
            auto const i1 = k;
            auto const i2 = k + stage_length;

            auto const x1 = std::complex{xre[i1], xim[i1]};
            auto const x2 = std::complex{xre[i2], xim[i2]};

            auto const xn1 = x1 + x2;
            xre[i1]        = xn1.real();
            xim[i1]        = xn1.imag();

            auto const xn2 = x1 - x2;
            xre[i2]        = xn2.real();
            xim[i2]        = xn2.imag();
        }

        auto const tw_re = stdex::submdspan(_tw.to_mdspan(), 0, stdex::full_extent);
        auto const tw_im = stdex::submdspan(_tw.to_mdspan(), 1, stdex::full_extent);
        stage_n(xre, xim, tw_re, tw_im, 1);
    }

    auto stage_n(
        inout_vector_of<Float> auto xre,
        inout_vector_of<Float> auto xim,
        in_vector_of<Float> auto tw_re,
        in_vector_of<Float> auto tw_im,
        int stage
    ) -> void
    {
        auto const log2_size    = static_cast<int>(order());
        auto const size         = 1 << log2_size;
        auto const stage_length = ipow<2>(stage);
        auto const stride       = ipow<2>(stage + 1);
        auto const tw_stride    = ipow<2>(log2_size - stage - 1);

        for (auto k{0}; k < size; k += stride) {
            for (auto pair{0}; pair < stage_length; ++pair) {
                auto const twi = pair * tw_stride;
                auto const tw  = std::complex{tw_re[twi], tw_im[twi]};

                auto const i1 = k + pair;
                auto const i2 = k + pair + stage_length;

                auto const x1 = std::complex{xre[i1], xim[i1]};
                auto const x2 = std::complex{xre[i2], xim[i2]};

                auto const xn1 = x1 + tw * x2;
                xre[i1]        = xn1.real();
                xim[i1]        = xn1.imag();

                auto const xn2 = x1 - tw * x2;
                xre[i2]        = xn2.real();
                xim[i2]        = xn2.imag();
            }
        }

        if (stage + 1 < log2_size) {
            stage_n(xre, xim, tw_re, tw_im, stage + 1);
        }
    }

    [[nodiscard]] static auto make_twiddles(size_type n)
    {
        auto tw = detail::make_radix2_twiddles<std::complex<Float>>(n, direction::forward);

        auto tw_buf = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{2, n};
        auto tw_re  = stdex::submdspan(tw_buf.to_mdspan(), 0, stdex::full_extent);
        auto tw_im  = stdex::submdspan(tw_buf.to_mdspan(), 1, stdex::full_extent);

        for (auto i{0U}; i < tw.extent(0); ++i) {
            tw_re[i] = tw(i).real();
            tw_im[i] = tw(i).imag();
        }

        return tw_buf;
    }

    fft::order _order;
    bitrevorder_plan _reorder{static_cast<size_t>(_order)};
    stdex::mdarray<Float, stdex::dextents<size_t, 2>> _tw{make_twiddles(size())};
};

}  // namespace neo::fft
