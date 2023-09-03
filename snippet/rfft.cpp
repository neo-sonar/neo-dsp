#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numbers>
#include <span>
#include <vector>

void fft(double* data, int n, int isign)
{
    int nn, mmax, m, j, istep, i;
    double wtemp, wr, wpr, wpi, wi, theta, tempr, tempi;

    nn = n << 1;
    j  = 1;
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

    mmax = 2;
    while (nn > mmax) {
        istep = mmax << 1;
        theta = isign * (6.28318530717959 / mmax);
        wtemp = std::sin(0.5 * theta);
        wpr   = -2.0 * wtemp * wtemp;
        wpi   = std::sin(theta);
        wr    = 1.0;
        wi    = 0.0;
        for (m = 1; m < mmax; m += 2) {
            for (i = m; i <= nn; i += istep) {
                j           = i + mmax;
                tempr       = wr * data[j - 1] - wi * data[j];
                tempi       = wr * data[j] + wi * data[j - 1];
                data[j - 1] = data[i - 1] - tempr;
                data[j]     = data[i] - tempi;
                data[i - 1] += tempr;
                data[i] += tempi;
            }
            wr = (wtemp = wr) * wpr - wi * wpi + wr;
            wi = wi * wpr + wtemp * wpi + wi;
        }
        mmax = istep;
    }
}

void fft(float* data, int n, int isign)
{
    int nn, mmax, m, j, istep, i;
    float wtemp, wr, wpr, wpi, wi, theta, tempr, tempi;

    nn = n << 1;
    j  = 1;
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

    mmax = 2;
    while (nn > mmax) {
        istep = mmax << 1;
        theta = isign * (6.28318530717959F / mmax);
        wtemp = std::sin(0.5F * theta);
        wpr   = -2.0F * wtemp * wtemp;
        wpi   = std::sin(theta);
        wr    = 1.0F;
        wi    = 0.0F;
        for (m = 1; m < mmax; m += 2) {
            for (i = m; i <= nn; i += istep) {
                j           = i + mmax;
                tempr       = wr * data[j - 1] - wi * data[j];
                tempi       = wr * data[j] + wi * data[j - 1];
                data[j - 1] = data[i - 1] - tempr;
                data[j]     = data[i] - tempi;
                data[i - 1] += tempr;
                data[i] += tempi;
            }
            wr = (wtemp = wr) * wpr - wi * wpi + wr;
            wi = wi * wpr + wtemp * wpi + wi;
        }
        mmax = istep;
    }
}

void fft(std::vector<double>& data, int isign) { fft(&data[0], data.size() / 2, isign); }

void fft(std::vector<std::complex<double>>& data, int isign) { fft((double*)(&data[0]), data.size(), isign); }

void fft(std::vector<float>& data, int isign) { fft(&data[0], data.size() / 2, isign); }

void fft(std::vector<std::complex<float>>& data, int isign) { fft((float*)(&data[0]), data.size(), isign); }

void rfft(std::vector<double>& data, int isign)
{
    auto const n  = data.size();
    auto const c1 = 0.5;
    auto const c2 = isign == 1 ? -0.5 : 0.5;

    double h1r;
    double h1i;
    double h2r;
    double h2i;
    double wr;
    double wi;
    double wtemp;
    double theta = std::numbers::pi / static_cast<double>(n >> 1);

    if (isign == 1) {
        fft(data, 1);
    } else {
        theta = -theta;
    }

    wtemp          = std::sin(0.5 * theta);
    auto const wpr = -2.0 * wtemp * wtemp;
    auto const wpi = std::sin(theta);
    wr             = 1.0 + wpr;
    wi             = wpi;

    for (auto i = 1; i < (n >> 2); i++) {
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
    if (isign == 1) {
        data[0] = (h1r = data[0]) + data[1];
        data[1] = h1r - data[1];
    } else {
        data[0] = c1 * ((h1r = data[0]) + data[1]);
        data[1] = c1 * (h1r - data[1]);
        fft(data, -1);
    }
}

void rfft(std::vector<float>& data, int isign)
{
    auto const n  = data.size();
    auto const c1 = 0.5F;
    auto const c2 = isign == 1 ? -0.5F : 0.5F;

    float h1r;
    float h1i;
    float h2r;
    float h2i;
    float wr;
    float wi;
    float wtemp;
    float theta = float(std::numbers::pi) / static_cast<float>(n >> 1);

    if (isign == 1) {
        fft(data, 1);
    } else {
        theta = -theta;
    }

    wtemp          = std::sin(0.5F * theta);
    auto const wpr = -2.0F * wtemp * wtemp;
    auto const wpi = std::sin(theta);
    wr             = 1.0F + wpr;
    wi             = wpi;

    for (auto i = 1; i < (n >> 2); i++) {
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
    if (isign == 1) {
        data[0] = (h1r = data[0]) + data[1];
        data[1] = h1r - data[1];
    } else {
        data[0] = c1 * ((h1r = data[0]) + data[1]);
        data[1] = c1 * (h1r - data[1]);
        fft(data, -1);
    }
}

auto main() -> int
{

    for (auto i{0}; i < 1000; ++i) {
        auto buffer = std::vector(4096UL, 0.0F);
        buffer[0]   = 1.0F;

        auto const start = std::chrono::system_clock::now();
        rfft(buffer, 1);
        rfft(buffer, -1);
        auto const stop    = std::chrono::system_clock::now();
        auto const elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(stop - start);
        auto const size    = static_cast<double>(buffer.size());
        auto const mflops  = static_cast<int>(std::lround(5.0 * size * std::log2(size) / elapsed.count()));

        std::printf(
            "Elapsed: %.1f MFLOPS: %d (%.1f)\n",
            elapsed.count(),
            mflops,
            buffer[0] * (2.0 / static_cast<double>(buffer.size()))
        );
    }

    auto buffer = std::vector(16UL, 0.0);
    buffer[0]   = 1.0;

    // rfft
    rfft(buffer, 1);
    for (auto val :
         std::span<std::complex<double>>{reinterpret_cast<std::complex<double>*>(buffer.data()), buffer.size() / 2U}) {
        std::printf("(%.1f, %.1f) ", val.real(), val.imag());
    }
    std::printf("\n");

    // irfft
    rfft(buffer, -1);
    for (auto val : buffer) {
        std::printf("%.1f ", val * (2.0 / static_cast<double>(buffer.size())));
    }
    std::printf("\n");

    return 0;
}
