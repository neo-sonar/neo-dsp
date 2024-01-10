#include <array>
#include <complex>
#include <concepts>
#include <cstdio>
#include <numbers>
#include <span>
#include <vector>

template<std::size_t Radix>
auto digitreversal(std::span<std::complex<double>> x, std::integral auto log2length) -> void
{
    auto const length = x.size();
    auto const n1     = (log2length % 2) == 0 ? static_cast<std::size_t>(std::sqrt(length))
                                              : static_cast<std::size_t>(std::sqrt(length / Radix));

    auto reverse = std::vector<std::size_t>(n1, std::size_t(0));
    reverse[1]   = length / Radix;

    // algorithm 2: compute seed table
    for (auto j{1ULL}; j < Radix; ++j) {
        reverse[j] = reverse[j - 1] + reverse[1];
        for (auto i{1ULL}; i < n1 / Radix; ++i) {
            reverse[Radix * i] = reverse[i] / Radix;
            for (auto k{1ULL}; k < Radix; ++k) {
                reverse[Radix * i + k] = reverse[Radix * i] + reverse[k];
            }
        }
    }

    // algorithm 1
    for (auto i{0ULL}; i < n1 - 1; ++i) {
        for (auto j = i + 1; j < n1; ++j) {
            auto const u = i + reverse[j];
            auto const v = j + reverse[i];
            std::swap(x[u], x[v]);

            if (log2length % 2 == 1) {
                for (auto z{1ULL}; z < Radix; ++z) {
                    auto const uu = i + reverse[j] + (z * n1);
                    auto const vv = j + reverse[i] + (z * n1);
                    std::swap(x[uu], x[vv]);
                }
            }
        }
    }
}

auto c2c_dit4(std::span<std::complex<double>> x, std::span<std::complex<double>> twiddle, std::size_t log2length)
    -> void
{
    auto const z = std::complex<double>{0.0, 1.0};

    auto length = 4UL;
    auto tss    = static_cast<std::size_t>(std::pow(4.0, log2length - 1UL));
    auto krange = 1UL;
    auto block  = x.size() / 4UL;
    auto base   = 0UL;

    for (auto w{0ULL}; w < log2length; ++w) {
        for (auto h{0ULL}; h < block; ++h) {
            for (auto k{0ULL}; k < krange; ++k) {
                auto const offset = length / 4;
                auto const avar   = base + k;
                auto const bvar   = base + k + offset;
                auto const cvar   = base + k + (2 * offset);
                auto const dvar   = base + k + (3 * offset);

                auto xbr1 = std::complex<double>{};
                auto xcr2 = std::complex<double>{};
                auto xdr3 = std::complex<double>{};
                if (k == 0) {
                    xbr1 = x[bvar];
                    xcr2 = x[cvar];
                    xdr3 = x[dvar];
                } else {
                    auto r1var = twiddle[k * tss];
                    auto r2var = twiddle[2 * k * tss];
                    auto r3var = twiddle[3 * k * tss];
                    xbr1       = (x[bvar] * r1var);
                    xcr2       = (x[cvar] * r2var);
                    xdr3       = (x[dvar] * r3var);
                }

                auto const evar = x[avar] + xcr2;
                auto const fvar = x[avar] - xcr2;
                auto const gvar = xbr1 + xdr3;
                auto const h    = xbr1 - xdr3;
                auto const j_h  = z * h;

                x[avar] = evar + gvar;
                x[bvar] = fvar - j_h;
                x[cvar] = -gvar + evar;
                x[dvar] = j_h + fvar;
            }

            base = base + (4UL * krange);
        }

        block  = block / 4UL;
        length = 4 * length;
        krange = 4 * krange;
        base   = 0;
        tss    = tss / 4;
    }
}

auto c2c_dif4(std::span<std::complex<double>> x, std::span<std::complex<double>> twiddle, std::size_t log2length)
    -> void
{
    auto const z = std::complex<double>{0.0, 1.0};

    auto length = static_cast<std::size_t>(std::pow(4.0, log2length));
    auto tss    = 1UL;
    auto krange = length / 4UL;
    auto block  = 1UL;
    auto base   = 0UL;

    for (auto w{0ULL}; w < log2length; ++w) {
        for (auto h{0ULL}; h < block; ++h) {
            for (auto k{0ULL}; k < krange; ++k) {
                auto const offset = length / 4UL;
                auto const a      = base + k;
                auto const b      = base + k + offset;
                auto const c      = base + k + (2 * offset);
                auto const d      = base + k + (3 * offset);
                auto const apc    = x[a] + x[c];
                auto const bpd    = x[b] + x[d];
                auto const amc    = x[a] - x[c];
                auto const bmd    = x[b] - x[d];
                x[a]              = apc + bpd;

                if (k == 0) {
                    x[b] = amc - (z * bmd);
                    x[c] = apc - bpd;
                    x[d] = amc + (z * bmd);
                } else {
                    auto r1 = twiddle[k * tss];
                    auto r2 = twiddle[2 * k * tss];
                    auto r3 = twiddle[3 * k * tss];
                    x[b]    = (amc - (z * bmd)) * r1;
                    x[c]    = (apc - bpd) * r2;
                    x[d]    = (amc + (z * bmd)) * r3;
                }
            }

            base = base + (4UL * krange);
        }

        block  = block * 4UL;
        length = length / 4UL;
        krange = krange / 4UL;
        base   = 0;
        tss    = tss * 4;
    }
}

auto main() -> int
{
    auto print = [](auto const& buffer, auto scale) {
        for (auto const z : buffer) {
            std::printf("%f,%f\n", z.real() * scale, z.imag() * scale);
        }

        std::puts("");
    };

    auto const log2size = 2UL;
    auto const n        = static_cast<size_t>(std::pow(4.0, log2size));

    auto buffer = std::vector<std::complex<double>>(n);
    buffer[0]   = std::complex<double>{1.0, 0.0};
    buffer[1]   = std::complex<double>{2.0, 0.0};
    buffer[2]   = std::complex<double>{3.0, 0.0};

    std::printf("n = %zu, log2len= %zu\n", n, log2size);

    auto kmax         = 3UL * (n / 4UL - 1UL);
    auto w_fwd        = std::vector<std::complex<double>>(kmax + 1);
    auto w_bwd        = std::vector<std::complex<double>>(kmax + 1);
    auto const two_pi = 2.0 * std::numbers::pi;
    for (auto i{0}; i < kmax + 1; ++i) {
        w_fwd[i] = std::polar(1.0, -1.0 * two_pi * double(i) / double(n));
        w_bwd[i] = std::polar(1.0, +1.0 * two_pi * double(i) / double(n));
    }

    print(buffer, 1.0);

    digitreversal<4>(buffer, log2size);
    c2c_dit4(buffer, w_fwd, log2size);

    print(buffer, 1.0);

    // digitreversal<4>(buffer, log2size);
    // c2c_dit4(buffer, w_bwd, log2size);

    // print(buffer, 1.0 / double(n));

    return 0;
}
