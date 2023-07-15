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
auto timeit(std::string_view name, size_t sizeOfT, size_t N, Func func)
{
    using microseconds = std::chrono::duration<double, std::micro>;

    auto const size = N;
    auto all_runs   = std::vector<double>{};

    func();
    func();
    func();

    for (auto i{0}; i < 10'000; ++i) {
        auto start = std::chrono::system_clock::now();
        func();
        auto stop = std::chrono::system_clock::now();

        all_runs.push_back(std::chrono::duration_cast<microseconds>(stop - start).count());
    }

    auto const runs            = std::span<double>(all_runs).subspan(500, all_runs.size() - 500 * 2);
    auto const dsize           = static_cast<double>(size);
    auto const avg             = std::reduce(runs.begin(), runs.end(), 0.0) / double(runs.size());
    auto const itemsPerSec     = static_cast<int>(std::lround(double(size) / avg));
    auto const megaBytesPerSec = std::round(double(size * sizeOfT) / avg) / 1000.0;

    std::printf("%-32s avg: %.1fus - N/usec: %d - GB/sec: %.2f\n", name.data(), avg, itemsPerSec, megaBytesPerSec);
}

}  // namespace neo::fft
