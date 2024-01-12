// SPDX-License-Identifier: MIT

#include <neo/fft.hpp>

#include <neo/testing/testing.hpp>

#include <benchmark/benchmark.h>

namespace {

template<typename ValueType>
auto copy(benchmark::State& state) -> void
{
    auto const size = static_cast<std::size_t>(state.range(0));

    auto src = std::vector<ValueType>(size);
    auto dst = std::vector<ValueType>(size);
    std::fill(src.begin(), src.end(), ValueType{1});

    for (auto _ : state) {
        std::copy(src.begin(), src.end(), dst.begin());
        benchmark::DoNotOptimize(dst[0]);
        benchmark::ClobberMemory();
    }

    auto const items = int64_t(state.iterations()) * int64_t(size);
    state.SetItemsProcessed(items);
    state.SetBytesProcessed(items * sizeof(ValueType));
}

}  // namespace

BENCHMARK(copy<char>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);
BENCHMARK(copy<int>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);
BENCHMARK(copy<float>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);
BENCHMARK(copy<std::complex<float>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 24);

BENCHMARK_MAIN();
