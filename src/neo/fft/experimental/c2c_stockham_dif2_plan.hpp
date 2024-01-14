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

/// http://wwwa.pikara.ne.jp/okojisan/otfft-en/stockham2.html
template<complex Complex>
struct c2c_stockham_dif2_plan_v1
{
    using value_type = Complex;
    using size_type  = std::size_t;

    c2c_stockham_dif2_plan_v1(from_order_tag /*tag*/, size_type order) : _order{order} {}

    [[nodiscard]] static constexpr auto max_order() noexcept -> size_type { return size_type{20}; }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type { return ipow<size_type(2)>(max_order()); }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return ipow<size_type(2)>(order()); }

    template<inout_vector_of<Complex> Vec>
    auto operator()(Vec x, direction dir) noexcept -> void
    {
        if (dir == direction::forward) {
            return fft(x);
        }
        return ifft(x);
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

    size_type _order;
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>> _w{make_twiddle_lut(size())};
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>> _work{size()};
};

/// Chapter 2.5
/// Fast Fourier Transform Algorithms for Parallel Computers
/// Daisuke Takahashi (2019)
/// ISBN 978-981-13-9964-0
template<complex Complex>
struct c2c_stockham_dif2_plan_v2
{
    using value_type = Complex;
    using size_type  = std::size_t;

    c2c_stockham_dif2_plan_v2(from_order_tag /*tag*/, size_type order) : _order{order} {}

    [[nodiscard]] static constexpr auto max_order() noexcept -> size_type { return size_type{20}; }

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type { return ipow<size_type(2)>(max_order()); }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return ipow<size_type(2)>(order()); }

    template<inout_vector_of<Complex> Vec>
    auto operator()(Vec x, direction dir) noexcept -> void
    {
        auto const n = size();
        auto const p = static_cast<size_t>(_order);
        auto const y = _work.to_mdspan();
        auto const w = dir == direction::forward ? stdex::submdspan(_w.to_mdspan(), 0, stdex::full_extent)
                                                 : stdex::submdspan(_w.to_mdspan(), 1, stdex::full_extent);

        auto l = n / 2U;
        auto m = 1U;

        for (auto t{1U}; t <= p; ++t) {
            copy(x, y);

            for (auto j{0U}; j < l; ++j) {
                auto const w1 = w[j * m];

                for (auto k{0U}; k < m; ++k) {
                    auto const c0 = y[k + (j * m) + (0 * l * m)];
                    auto const c1 = y[k + (j * m) + (1 * l * m)];

                    x[k + (2 * j * m) + (0 * m)] = c0 + c1;
                    x[k + (2 * j * m) + (1 * m)] = (c0 - c1) * w1;
                }
            }

            l = l / 2;
            m = m * 2;
        }
    }

private:
    static auto make_twiddle_lut(size_t n) -> stdex::mdarray<Complex, stdex::dextents<std::size_t, 2>>
    {
        auto w = stdex::mdarray<Complex, stdex::dextents<std::size_t, 2>>{2, n / 2};
        for (std::size_t i = 0; i < w.extent(1); i++) {
            w(0, i) = twiddle<Complex>(n, i, fft::direction::backward);
            w(1, i) = twiddle<Complex>(n, i, fft::direction::forward);
        }
        return w;
    }

    size_type _order;
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 2>> _w{make_twiddle_lut(size())};
    stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>> _work{size()};
};

}  // namespace neo::fft::experimental
