#pragma once

#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>

#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

namespace neo::fft::experimental {

template<inout_vector Vec>
    requires(std::floating_point<typename Vec::value_type>)
constexpr auto bit_reverse_permutation(Vec x) -> void
{
    auto const nn = static_cast<int>(x.extent(0));
    auto const n  = nn / 2;

    auto j = 1;
    for (auto i{1}; i < nn; i += 2) {
        if (j > i) {
            std::swap(x[j - 1], x[i - 1]);
            std::swap(x[j], x[i]);
        }
        auto m = n;
        while (m >= 2 && j > m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

template<neo::inout_vector Vec>
    requires(std::floating_point<typename Vec::value_type>)
void fft(Vec x, direction dir)
{
    using Float = typename Vec::value_type;

    auto const nn   = static_cast<int>(x.extent(0));
    auto const sign = dir == direction::forward ? Float(1) : Float(-1);

    bit_reverse_permutation(x);

    auto mmax = 2;
    while (nn > mmax) {
        auto const step  = mmax << 1;
        auto const theta = sign * (Float(6.28318530717959) / Float(mmax));

        auto wtemp = std::sin(Float(0.5) * theta);
        auto wpr   = Float(-2) * wtemp * wtemp;
        auto wpi   = std::sin(theta);
        auto wr    = Float(1);
        auto wi    = Float(0);

        for (auto m{1}; m < mmax; m += 2) {
            for (auto i{m}; i <= nn; i += step) {
                auto const j   = i + mmax;
                auto const jn1 = j - 1;
                auto const in1 = i - 1;

                auto const tempr = wr * x[jn1] - wi * x[j];
                auto const tempi = wr * x[j] + wi * x[jn1];

                x[jn1] = x[in1] - tempr;
                x[j]   = x[i] - tempi;

                x[in1] += tempr;
                x[i] += tempi;
            }

            wtemp = wr;
            wr    = wtemp * wpr - wi * wpi + wr;
            wi    = wi * wpr + wtemp * wpi + wi;
        }

        mmax = step;
    }
}

template<inout_vector Vec>
auto rfft(Vec x, direction dir) -> void
{
    using Float = typename Vec::value_type;

    auto const n     = x.extent(0);
    auto const c1    = Float(0.5);
    auto const c2    = dir == direction::forward ? Float(-0.5) : Float(0.5);
    auto const theta = [=] {
        auto const t = static_cast<Float>(std::numbers::pi) / static_cast<Float>(n >> 1);
        return dir == direction::forward ? t : -t;
    }();

    if (dir == direction::forward) {
        fft(x, direction::forward);
    }

    auto wtemp     = std::sin(Float(0.5) * theta);
    auto const wpr = Float(-2) * wtemp * wtemp;
    auto const wpi = std::sin(theta);
    auto wr        = Float(1) + wpr;
    auto wi        = wpi;

    for (auto i = 1U; i < (n >> 2); i++) {
        auto const i1 = i + i;
        auto const i2 = i1 + 1;
        auto const i3 = n - i1;
        auto const i4 = i3 + 1;

        auto const h1r = c1 * (x[i1] + x[i3]);
        auto const h1i = c1 * (x[i2] - x[i4]);
        auto const h2r = -c2 * (x[i2] + x[i4]);
        auto const h2i = c2 * (x[i1] - x[i3]);

        x[i1] = h1r + wr * h2r - wi * h2i;
        x[i2] = h1i + wr * h2i + wi * h2r;
        x[i3] = h1r - wr * h2r + wi * h2i;
        x[i4] = -h1i + wr * h2i + wi * h2r;

        auto const tmp = wr;
        wr             = tmp * wpr - wi * wpi + wr;
        wi             = wi * wpr + tmp * wpi + wi;
    }

    auto const h1r = x[0];
    if (dir == direction::forward) {
        x[0] = h1r + x[1];
        x[1] = h1r - x[1];
    } else {
        x[0] = c1 * (h1r + x[1]);
        x[1] = c1 * (h1r - x[1]);
        fft(x, direction::backward);
    }
}

}  // namespace neo::fft::experimental
