#pragma once

#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>

#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

namespace neo::fft::experimental {

template<neo::inout_vector Vec>
    requires(std::floating_point<typename Vec::value_type>)
void fft(Vec data, int n, int sign)
{
    using Float = typename Vec::value_type;

    int nn, mmax, m, istep, i;
    Float wtemp, wr, wpr, wpi, wi, theta;

    nn = n << 1;

    {
        auto j = 1;
        for (i = 1; i < nn; i += 2) {
            if (j > i) {
                std::swap(data[j - 1], data[i - 1]);
                std::swap(data[j], data[i]);
            }
            m = n;
            while (m >= 2 && j > m) {
                j -= m;
                m >>= 1;
            }
            j += m;
        }
    }

    mmax = 2;
    while (nn > mmax) {
        istep = mmax << 1;
        theta = Float(sign) * (Float(6.28318530717959) / Float(mmax));
        wtemp = std::sin(Float(0.5) * theta);
        wpr   = Float(-2) * wtemp * wtemp;
        wpi   = std::sin(theta);
        wr    = Float(1);
        wi    = Float(0);
        for (m = 1; m < mmax; m += 2) {
            for (i = m; i <= nn; i += istep) {
                auto const j   = i + mmax;
                auto const jn1 = j - 1;
                auto const in1 = i - 1;

                auto const tempr = wr * data[jn1] - wi * data[j];
                auto const tempi = wr * data[j] + wi * data[jn1];

                data[jn1] = data[in1] - tempr;
                data[j]   = data[i] - tempi;

                data[in1] += tempr;
                data[i] += tempi;
            }

            wr = (wtemp = wr) * wpr - wi * wpi + wr;
            wi = wi * wpr + wtemp * wpi + wi;
        }
        mmax = istep;
    }
}

template<std::floating_point Float>
inline void fft(std::vector<Float>& data, int sign)
{
    auto const x = stdex::mdspan{&data[0], stdex::extents{data.size()}};
    fft(x, static_cast<int>(data.size() / 2), sign);
}

template<std::floating_point Float>
inline void rfft(std::vector<Float>& data, int sign)
{
    auto const n  = data.size();
    auto const c1 = Float(0.5);
    auto const c2 = sign == 1 ? Float(-0.5) : Float(0.5);

    Float h1r;
    Float h1i;
    Float h2r;
    Float h2i;
    Float wr;
    Float wi;
    Float wtemp;
    Float theta = Float(std::numbers::pi) / static_cast<Float>(n >> 1);

    if (sign == 1) {
        fft(data, 1);
    } else {
        theta = -theta;
    }

    wtemp          = std::sin(Float(0.5) * theta);
    auto const wpr = Float(-2) * wtemp * wtemp;
    auto const wpi = std::sin(theta);
    wr             = Float(1) + wpr;
    wi             = wpi;

    for (auto i = 1U; i < (n >> 2); i++) {
        auto const i1 = i + i;
        auto const i2 = 1 + i1;
        auto const i3 = n - i1;
        auto const i4 = 1 + i3;

        h1r = c1 * (data[i1] + data[i3]);
        h1i = c1 * (data[i2] - data[i4]);
        h2r = -c2 * (data[i2] + data[i4]);
        h2i = c2 * (data[i1] - data[i3]);

        data[i1] = h1r + wr * h2r - wi * h2i;
        data[i2] = h1i + wr * h2i + wi * h2r;
        data[i3] = h1r - wr * h2r + wi * h2i;
        data[i4] = -h1i + wr * h2i + wi * h2r;

        wr = (wtemp = wr) * wpr - wi * wpi + wr;
        wi = wi * wpr + wtemp * wpi + wi;
    }
    if (sign == 1) {
        data[0] = (h1r = data[0]) + data[1];
        data[1] = h1r - data[1];
    } else {
        data[0] = c1 * ((h1r = data[0]) + data[1]);
        data[1] = c1 * (h1r - data[1]);
        fft(data, -1);
    }
}

}  // namespace neo::fft::experimental
