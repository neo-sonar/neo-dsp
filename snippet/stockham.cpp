#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <numbers>
#include <span>
#include <vector>

// typedef std::complex<double> complex_t;

// // n  : sequence length
// // s  : stride
// // eo : x is output if eo == 0, y is output if eo == 1
// // x  : input sequence(or output sequence if eo == 0)
// // y  : work area(or output sequence if eo == 1)
// void fft0(int n, int s, bool eo, complex_t* x, complex_t* y)
// {
//     int const m         = n / 2;
//     double const theta0 = 2 * std::numbers::pi / n;

// if (n == 2) {
//     complex_t* z = eo ? y : x;
//     for (int q = 0; q < s; q++) {
//         const complex_t a = x[q + 0];
//         const complex_t b = x[q + s];
//         z[q + 0]          = a + b;
//         z[q + s]          = a - b;
//     }
// } else if (n >= 4) {
//     for (int p = 0; p < m; p++) {
//         const complex_t wp = complex_t(cos(p * theta0), -sin(p * theta0));
//         for (int q = 0; q < s; q++) {
//             const complex_t a      = x[q + s * (p + 0)];
//             const complex_t b      = x[q + s * (p + m)];
//             y[q + s * (2 * p + 0)] = a + b;
//             y[q + s * (2 * p + 1)] = (a - b) * wp;
//         }
//     }
//     fft0(n / 2, 2 * s, !eo, y, x);
// }
// }

// struct stockham_plan
// {
//     explicit stockham_plan(int size) : _work(static_cast<std::size_t>(size), complex_t{}) {}

// // Fourier transform
// // n : sequence length
// // x : input/output sequence
// auto fft(std::span<complex_t> x) -> void
// {
//     assert(std::cmp_equal(x.size(), _work.size()));
//     fft0(x.size(), 1, 0, x.data(), _work.data());
// }

// // Inverse Fourier transform
// // n : sequence length
// // x : input/output sequence
// auto ifft(std::span<complex_t> x) -> void
// {
//     assert(std::cmp_equal(x.size(), _work.size()));

// auto const n = static_cast<int>(x.size());

// for (int p = 0; p < n; p++) {
//     x[p] = std::conj(x[p]);
// }

// fft0(x.size(), 1, 0, x.data(), _work.data());

// for (int k = 0; k < n; k++) {
//     x[k] = std::conj(x[k] / double(n));
// }
// }

// private:
//     std::vector<complex_t> _work;
// };

// // auto main() -> int
// // {
// //     complex_t buffer[4]{};
// //     buffer[0] = {1.0, 0.0};
// //     auto plan = stockham_plan{static_cast<int>(std::size(buffer))};

// // plan.fft(buffer);
// // for (auto val : buffer) {
// //     std::cout << val << '\n';
// // }
// // std::cout << '\n';
// // plan.ifft(buffer);
// // for (auto val : buffer) {
// //     std::cout << val << '\n';
// // }
// // return 0;
// // }

#include <immintrin.h>

#include <cmath>
#include <complex>
#include <iostream>

struct complex_t
{
    double Re, Im;
};

__m256d mulpz2(const __m256d ab,
               const __m256d xy)  // Multiplication of complex numbers
{
    const __m256d aa = _mm256_unpacklo_pd(ab, ab);
    const __m256d bb = _mm256_unpackhi_pd(ab, ab);
    const __m256d yx = _mm256_shuffle_pd(xy, xy, 5);
    return _mm256_addsub_pd(_mm256_mul_pd(aa, xy), _mm256_mul_pd(bb, yx));
}

void fft0(int n, int s, bool eo, complex_t* x, complex_t* y)
// n  : sequence length
// s  : stride
// eo : x is output if eo == 0, y is output if eo == 1
// x  : input sequence(or output sequence if eo == 0)
// y  : work area(or output sequence if eo == 1)
{
    int const m         = n / 2;
    double const theta0 = 2 * std::numbers::pi / n;

    if (n == 2) {
        complex_t* z = eo ? y : x;
        if (s == 1) {
            double* xd      = &x->Re;
            double* zd      = &z->Re;
            const __m128d a = _mm_load_pd(xd + 2 * 0);
            const __m128d b = _mm_load_pd(xd + 2 * 1);
            _mm_store_pd(zd + 2 * 0, _mm_add_pd(a, b));
            _mm_store_pd(zd + 2 * 1, _mm_sub_pd(a, b));
        } else {
            for (int q = 0; q < s; q += 2) {
                double* xd      = &(x + q)->Re;
                double* zd      = &(z + q)->Re;
                const __m256d a = _mm256_load_pd(xd + 2 * 0);
                const __m256d b = _mm256_load_pd(xd + 2 * s);
                _mm256_store_pd(zd + 2 * 0, _mm256_add_pd(a, b));
                _mm256_store_pd(zd + 2 * s, _mm256_sub_pd(a, b));
            }
        }
    } else if (n >= 4) {
        if (s == 1) {
            for (int p = 0; p < m; p += 2) {
                auto const p0    = std::polar(1.0, (p + 0) * theta0);
                auto const p1    = std::polar(1.0, (p + 1) * theta0);
                auto const cs0   = p0.real();
                auto const sn0   = p0.imag();
                auto const cs1   = p1.real();
                auto const sn1   = p1.imag();
                const __m256d wp = _mm256_setr_pd(cs0, -sn0, cs1, -sn1);
                double* xd       = &(x + p)->Re;
                double* yd       = &(y + 2 * p)->Re;
                const __m256d a  = _mm256_load_pd(xd + 2 * 0);
                const __m256d b  = _mm256_load_pd(xd + 2 * m);
                const __m256d aA = _mm256_add_pd(a, b);
                const __m256d bB = mulpz2(wp, _mm256_sub_pd(a, b));
                const __m256d ab = _mm256_permute2f128_pd(aA, bB, 0x20);
                const __m256d AB = _mm256_permute2f128_pd(aA, bB, 0x31);
                _mm256_store_pd(yd + 2 * 0, ab);
                _mm256_store_pd(yd + 2 * 2, AB);
            }
        } else {
            for (int p = 0; p < m; p++) {
                auto const p0    = std::polar(1.0, p * theta0);
                double const cs  = p0.real();
                double const sn  = p0.imag();
                const __m256d wp = _mm256_setr_pd(cs, -sn, cs, -sn);
                for (int q = 0; q < s; q += 2) {
                    double* xd      = &(x + q)->Re;
                    double* yd      = &(y + q)->Re;
                    const __m256d a = _mm256_load_pd(xd + 2 * s * (p + 0));
                    const __m256d b = _mm256_load_pd(xd + 2 * s * (p + m));
                    _mm256_store_pd(yd + 2 * s * (2 * p + 0), _mm256_add_pd(a, b));
                    _mm256_store_pd(yd + 2 * s * (2 * p + 1), mulpz2(wp, _mm256_sub_pd(a, b)));
                }
            }
        }
        fft0(n / 2, 2 * s, !eo, y, x);
    }
}

struct simd_plan
{
    explicit simd_plan(int size) : _y(static_cast<std::size_t>(size)), _z(static_cast<std::size_t>(size)) {}

    // n : sequence length
    // x : input/output sequence
    void fft(int n, std::complex<double>* x)  // Fourier transform
    {
        for (int p = 0; p < n; p++) {
            _y[p].Re = x[p].real();
            _y[p].Im = x[p].imag();
        }
        fft0(n, 1, 0, _y.data(), _z.data());
        for (int k = 0; k < n; k++)
            x[k] = std::complex<double>(_y[k].Re / n, _y[k].Im / n);
    }

    // n : sequence length
    // x : input/output sequence
    void ifft(int n, std::complex<double>* x)  // Inverse Fourier transform
    {
        for (int p = 0; p < n; p++) {
            _y[p].Re = x[p].real();
            _y[p].Im = -x[p].imag();
        }
        fft0(n, 1, 0, _y.data(), _z.data());
        for (int k = 0; k < n; k++)
            x[k] = std::complex<double>(_y[k].Re, -_y[k].Im);
    }

private:
    std::vector<complex_t> _y;
    std::vector<complex_t> _z;
};

auto main(int argc, char** argv) -> int
{
    auto N = 1024UL;
    if (argc == 2) {
        N = std::stoul(argv[1]);
    }

    auto runs = std::vector(75'000UL, 0);

    auto buffer = std::vector(N, std::complex<double>{});
    auto plan   = simd_plan{static_cast<int>(std::size(buffer))};
    buffer[0]   = 1.0F;

    for (auto i{0UL}; i < runs.size(); ++i) {

        auto const start = std::chrono::system_clock::now();
        plan.fft(std::ssize(buffer), buffer.data());
        // plan.ifft(std::ssize(buffer), buffer.data());
        auto const stop    = std::chrono::system_clock::now();
        auto const elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(stop - start);
        auto const size    = static_cast<double>(buffer.size());
        auto const mflops  = static_cast<int>(std::lround(5.0 * size * std::log2(size) / elapsed.count()));

        runs[i] = mflops;
    }

    std::printf(
        "MFLOPS: %d min, %d max\n",
        *std::min_element(runs.begin(), runs.end()),
        *std::max_element(runs.begin(), runs.end())
    );
}
