#pragma once

#include <neo/config.hpp>

#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/bitrevorder.hpp>
#include <neo/fft/direction.hpp>
#include <neo/fft/kernel/radix2.hpp>
#include <neo/fft/twiddle.hpp>

#include <cassert>
#include <utility>
#include <vector>

namespace neo::fft {

namespace detail {

template<typename Complex, int Order, int Stage>
struct static_dit2_stage
{
    auto operator()(inout_vector auto x, in_vector auto twiddles) -> void
        requires(Stage == 0)
    {
        static constexpr auto const size         = 1 << Order;
        static constexpr auto const stage_length = 1;  // ipow<2>(0)
        static constexpr auto const stride       = 2;  // ipow<2>(0 + 1)

        for (auto k{0}; k < static_cast<int>(size); k += stride) {
            auto const i1 = k;
            auto const i2 = k + stage_length;

            auto const temp = x[i1] + x[i2];
            x[i2]           = x[i1] - x[i2];
            x[i1]           = temp;
        }

        static_dit2_stage<Complex, Order, 1>{}(x, twiddles);
    }

    auto operator()(inout_vector auto x, in_vector auto twiddles) -> void
        requires(Stage != 0 and Stage < Order)
    {
        static constexpr auto const size         = 1 << Order;
        static constexpr auto const stage_length = ipow<2>(Stage);
        static constexpr auto const stride       = ipow<2>(Stage + 1);
        static constexpr auto const tw_stride    = ipow<2>(Order - Stage - 1);

        for (auto k{0}; k < size; k += stride) {
            for (auto pair{0}; pair < stage_length; ++pair) {
                auto const tw = twiddles[pair * tw_stride];

                auto const i1 = k + pair;
                auto const i2 = k + pair + stage_length;

                auto const temp = x[i1] + tw * x[i2];
                x[i2]           = x[i1] - tw * x[i2];
                x[i1]           = temp;
            }
        }

        static_dit2_stage<Complex, Order, Stage + 1>{}(x, twiddles);
    }

    auto operator()(inout_vector auto /*x*/, in_vector auto /*twiddles*/) -> void
        requires(Stage == Order)
    {}
};

}  // namespace detail

template<typename Complex, std::size_t Order>
struct static_fft_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    static_fft_plan() = default;

    [[nodiscard]] static constexpr auto size() -> std::size_t { return 1 << Order; }

    [[nodiscard]] static constexpr auto order() -> std::size_t { return Order; }

    template<inout_vector InOutVec>
        requires std::same_as<typename InOutVec::value_type, Complex>
    auto operator()(InOutVec x, direction dir) noexcept -> void
    {
        _reorder(x);

        if (dir == direction::forward) {
            detail::static_dit2_stage<Complex, Order, 0>{}(x, _wf.to_mdspan());
        } else {
            detail::static_dit2_stage<Complex, Order, 0>{}(x, _wb.to_mdspan());
        }
    }

private:
    bitrevorder_plan _reorder{order()};
    stdex::mdarray<Complex, stdex::extents<std::size_t, size() / 2>> _wf{
        make_radix2_twiddles<Complex, size()>(direction::forward),
    };
    stdex::mdarray<Complex, stdex::extents<std::size_t, size() / 2>> _wb{
        make_radix2_twiddles<Complex, size()>(direction::backward),
    };
};

}  // namespace neo::fft
