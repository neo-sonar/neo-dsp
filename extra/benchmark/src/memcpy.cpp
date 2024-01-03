// SPDX-License-Identifier: MIT

#include <neo/fft.hpp>

#include <neo/testing/testing.hpp>

#include <benchmark/benchmark.h>

namespace {

auto memcopy(benchmark::State& state) -> void
{
    auto src = std::make_unique<char[]>(state.range(0));
    auto dst = std::make_unique<char[]>(state.range(0));
    std::memset(src.get(), 'x', state.range(0));

    for (auto _ : state) {
        std::memcpy(dst.get(), src.get(), state.range(0));
        benchmark::DoNotOptimize(dst[0]);
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
}

}  // namespace

BENCHMARK(memcopy)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);

BENCHMARK_MAIN();
