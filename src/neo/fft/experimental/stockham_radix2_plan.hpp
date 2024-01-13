// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>
#include <neo/fft/order.hpp>
#include <neo/fft/twiddle.hpp>
#include <neo/math/conj.hpp>
#include <neo/math/ipow.hpp>

namespace neo::fft::experimental {

template<complex Complex>
struct stockham_radix2_plan
{
    using value_type = Complex;
    using size_type  = std::size_t;

    explicit stockham_radix2_plan(fft::order order) : _order{order} {}

    [[nodiscard]] static constexpr auto max_order() noexcept -> fft::order { return fft::order{27}; }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type
    {
        return ipow<size_type(2)>(static_cast<size_type>(max_order()));
    }

    [[nodiscard]] auto order() const noexcept -> fft::order { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type
    {
        return ipow<size_type(2)>(static_cast<size_type>(order()));
    }

    template<inout_vector_of<Complex> Vec>
    auto operator()(Vec x, direction dir) noexcept -> void
    {
        if (dir == direction::forward) {
            return fft(x);
        } else {
            return ifft(x);
        }
    }

private:
    auto fft(inout_vector_of<Complex> auto x) noexcept -> void
    {
        fft0(size(), 1, false, x, _work.to_mdspan(), _w.to_mdspan());
    }

    void ifft(inout_vector_of<Complex> auto x)
    {
        auto conjugate = [n = size()](inout_vector_of<Complex> auto vec) {
            for (std::size_t i = 0; i < n; ++i) {
                vec[i] = neo::math::conj(vec[i]);
            }
        };

        conjugate(x);
        fft0(size(), 1, false, x, _work.to_mdspan(), _w.to_mdspan());
        conjugate(x);
    }

    // n  : sequence length
    // s  : stride
    // eo : x is output if eo == 0, work is output if eo == 1
    // x  : input sequence(or output sequence if eo == 0)
    // work  : work area(or output sequence if eo == 1)
    static auto fft0(
        std::size_t n,
        std::size_t s,
        bool eo,
        inout_vector_of<Complex> auto x,
        inout_vector_of<Complex> auto work,
        in_vector_of<Complex> auto w
    ) -> void
    {
        if (n == 1) {
            if (eo) {
                for (std::size_t q = 0; q < s; q++) {
                    work[q] = x[q];
                }
            }
            return;
        }

        auto const m = n / 2U;

        for (std::size_t p = 0; p < m; p++) {
            auto wp = w[p * s];

            for (std::size_t q = 0; q < s; q++) {
                auto const a = x[q + s * (p + 0)];
                auto const b = x[q + s * (p + m)];

                work[q + s * (2 * p + 0)] = a + b;
                work[q + s * (2 * p + 1)] = (a - b) * wp;
            }
        }

        fft0(n / 2, 2 * s, !eo, work, x, w);
    }

    static auto make_twiddle_lut(size_t n) -> stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>>
    {
        auto w = stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>>(n / 2);
        for (std::size_t i = 0; i < w.size(); i++) {
            w(i) = twiddle<Complex>(n, i, fft::direction::backward);
        }
        return w;
    }

    fft::order _order;
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>> _w{make_twiddle_lut(size())};
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>> _work{size()};
};

}  // namespace neo::fft::experimental
