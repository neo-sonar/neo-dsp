#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <span>
#include <string_view>
#include <vector>

namespace neo::fft {

template<typename Func>
auto benchmark_fft(std::string_view name, size_t N, size_t multiplier, Func func)
{
    using microseconds = std::chrono::duration<double, std::micro>;

    auto const size = N * multiplier;
    auto all_runs   = std::vector<double>{};

    func();
    func();
    func();

    for (auto i{0}; i < 100'000; ++i) {
        auto start = std::chrono::system_clock::now();
        func();
        auto stop = std::chrono::system_clock::now();

        all_runs.push_back(std::chrono::duration_cast<microseconds>(stop - start).count());
    }

    auto const runs   = std::span<double>(all_runs).subspan(1000, all_runs.size() - 1000 * 2);
    auto const dsize  = static_cast<double>(size);
    auto const avg    = std::reduce(runs.begin(), runs.end(), 0.0) / double(runs.size());
    auto const mflops = static_cast<int>(std::lround(5.0 * dsize * std::log2(dsize) / avg));

    std::printf(
        "%-40s N: %-5zu - size: %-5zu - runs: %zu - avg: %.1fus - min: %.1fus - max: "
        "%.1fus - mflops: %d\n",
        name.data(),
        N,
        size,
        std::size(all_runs),
        avg,
        *std::min_element(runs.begin(), runs.end()),
        *std::max_element(runs.begin(), runs.end()),
        mflops
    );
}

template<typename Func>
auto timeit(std::string_view name, size_t sizeOfT, size_t N, Func func)
{
    using microseconds = std::chrono::duration<double, std::micro>;

    auto const size       = N;
    auto const iterations = 25'000U;
    auto const margin     = iterations / 20U;

    auto all_runs = std::vector<double>(iterations);

    func();
    func();
    func();

    for (auto i{0U}; i < iterations; ++i) {
        auto start = std::chrono::system_clock::now();
        func();
        auto stop = std::chrono::system_clock::now();

        all_runs[i] = std::chrono::duration_cast<microseconds>(stop - start).count();
    }

    auto const runs            = std::span<double>(all_runs).subspan(margin, all_runs.size() - margin * 2);
    auto const avg             = std::reduce(runs.begin(), runs.end(), 0.0) / double(runs.size());
    auto const itemsPerSec     = static_cast<int>(std::lround(double(size) / avg));
    auto const megaBytesPerSec = std::round(double(size * sizeOfT) / avg) / 1000.0;
    std::printf("%-32s avg: %.1fus - GB/sec: %.2f - N/usec: %d\n", name.data(), avg, megaBytesPerSec, itemsPerSec);
}

}  // namespace neo::fft
