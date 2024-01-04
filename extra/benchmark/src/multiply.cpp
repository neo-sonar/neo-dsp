// SPDX-License-Identifier: MIT

#include <neo/algorithm.hpp>
#include <neo/fixed_point.hpp>

#include <neo/testing/testing.hpp>

#include <benchmark/benchmark.h>

namespace {

template<typename Type>
auto multiply(benchmark::State& state) -> void
{
    auto const size = state.range(0);
    auto const lhs  = neo::generate_noise_signal<Type>(size, std::random_device{}());
    auto const rhs  = neo::generate_noise_signal<Type>(size, std::random_device{}());

    auto out = stdex::mdarray<Type, stdex::dextents<size_t, 1>>{lhs.extents()};

    for (auto _ : state) {
        neo::multiply(lhs.to_mdspan(), rhs.to_mdspan(), out.to_mdspan());
        benchmark::DoNotOptimize(out(0));
        benchmark::ClobberMemory();
    }

    auto const items = static_cast<int64_t>(state.iterations()) * size;
    state.SetItemsProcessed(items);
    state.SetBytesProcessed(items * sizeof(Type));
}

}  // namespace

BENCHMARK(multiply<float>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);
BENCHMARK(multiply<double>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);
BENCHMARK(multiply<neo::q7>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);
BENCHMARK(multiply<neo::q15>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);
BENCHMARK(multiply<neo::fixed_point<int16_t, 14>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);
BENCHMARK_MAIN();
