// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>
#include <neo/fft/experimental/digitrevorder.hpp>
#include <neo/fft/order.hpp>
#include <neo/fft/twiddle.hpp>
#include <neo/math/ipow.hpp>

namespace neo::fft::experimental {

/// https://github.com/scientificgo/fft/blob/master/radix3.go
template<complex Complex>
struct c2c_dif3_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    explicit c2c_dif3_plan(from_order_tag /*tag*/, size_type order) : _order{order} {}

    [[nodiscard]] static constexpr auto max_order() noexcept -> size_type { return 17; }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type { return ipow<size_type(3)>(max_order()); }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return ipow<size_type(3)>(order()); }

    template<inout_vector Vec>
        requires std::same_as<typename Vec::value_type, Complex>
    auto operator()(Vec x, direction dir) noexcept -> void
    {
        using Float = value_type_t<Complex>;

        auto const w31 = twiddle<Complex>(3, 1, direction::forward);
        auto const w32 = twiddle<Complex>(3, 2, direction::forward);

        auto const radix = 3;
        auto const len   = static_cast<int>(x.extent(0));

        _reorder(x);

        for (auto m = radix; m <= len; m *= radix) {
            // Calculate twiddle factor.
            auto const w = twiddle<Complex>(m, 1, dir);

            auto mr = m / radix;
            auto wi = Complex{Float(1), Float(0)};

            for (auto i = 0; i < mr; i++) {
                for (auto j = 0; j < len; j += m) {
                    // Retrieve subset of points.
                    auto t0 = x[i + j];
                    auto t1 = x[i + j + mr];
                    auto t2 = x[i + j + 2 * mr];

                    // Apply twiddle factors w**(i+k) for 1 ≤ k < radix.
                    t1 *= wi;
                    t2 *= wi * wi;

                    // Transform points using radix-point DFT.
                    x[i + j] += t1 + t2;
                    if (dir == direction::forward) {
                        x[i + j + mr]     = t0 + t1 * w31 + t2 * w32;
                        x[i + j + 2 * mr] = t0 + t1 * w32 + t2 * w31;
                    } else {
                        // 1/w31 = w32, etc.
                        x[i + j + mr]     = t0 + t1 * w32 + t2 * w31;
                        x[i + j + 2 * mr] = t0 + t1 * w31 + t2 * w32;
                    }
                }
                wi *= w;
            }
        }
    }

private:
    size_type _order;
    digitrevorder_plan<3> _reorder{size()};
};

}  // namespace neo::fft::experimental
