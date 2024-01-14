// SPDX-License-Identifier: MIT

#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/complex/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>
#include <neo/fft/order.hpp>
#include <neo/fft/twiddle.hpp>
#include <neo/math/conj.hpp>
#include <neo/math/ipow.hpp>

namespace neo::fft::experimental {

/// Chapter 3.3
/// Fast Fourier Transform Algorithms for Parallel Computers
/// Daisuke Takahashi (2019)
/// ISBN 978-981-13-9964-0
template<complex Complex>
struct c2c_stockham_dif4_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    explicit c2c_stockham_dif4_plan(fft::order order) : _order{order} {}

    [[nodiscard]] static constexpr auto max_order() noexcept -> fft::order { return fft::order{11}; }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type
    {
        return ipow<size_type(4)>(static_cast<size_type>(max_order()));
    }

    [[nodiscard]] auto order() const noexcept -> fft::order { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type
    {
        return ipow<size_type(4)>(static_cast<size_type>(order()));
    }

    template<inout_vector_of<Complex> Vec>
    auto operator()(Vec x, direction dir) noexcept -> void
    {
        using Float = value_type_t<Complex>;

        auto const sign = dir == direction::forward ? Float(-1) : Float(+1);
        auto const I    = Complex{Float(0), sign};

        auto const n = size();
        auto const p = static_cast<size_t>(_order);
        auto const y = _work.to_mdspan();

        auto L = n / 4U;
        auto m = 1U;

        for (auto t{1U}; t <= p; ++t) {
            copy(x, y);  // TODO

            for (auto j{0U}; j < L; ++j) {
                auto const w1 = twiddle<Complex>(L * 4, 1 * j, dir);
                auto const w2 = twiddle<Complex>(L * 4, 2 * j, dir);
                auto const w3 = twiddle<Complex>(L * 4, 3 * j, dir);

                for (auto k{0U}; k < m; ++k) {
                    auto const c0 = y[k + (j * m) + (0 * L * m)];
                    auto const c1 = y[k + (j * m) + (1 * L * m)];
                    auto const c2 = y[k + (j * m) + (2 * L * m)];
                    auto const c3 = y[k + (j * m) + (3 * L * m)];

                    auto const d0 = c0 + c2;
                    auto const d1 = c0 - c2;
                    auto const d2 = c1 + c3;
                    auto const d3 = (c1 - c3) * I;

                    x[k + (4 * j * m) + (0 * m)] = d0 + d2;
                    x[k + (4 * j * m) + (1 * m)] = (d1 + d3) * w1;
                    x[k + (4 * j * m) + (2 * m)] = (d0 - d2) * w2;
                    x[k + (4 * j * m) + (3 * m)] = (d1 - d3) * w3;
                }
            }

            L = L / 4;
            m = m * 4;
        }
    }

private:
    fft::order _order;
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>> _work{size()};
};

}  // namespace neo::fft::experimental
