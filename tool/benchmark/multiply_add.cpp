#include <neo/algorithm.hpp>
#include <neo/fixed_point.hpp>

#include <neo/testing/testing.hpp>

#include <benchmark/benchmark.h>

namespace {

template<typename ValueType>
auto multiply_add(benchmark::State& state) -> void
{
    auto const size = static_cast<std::size_t>(state.range(0));

    auto const x_buf = neo::generate_noise_signal<ValueType>(size, std::random_device{}());
    auto const y_buf = neo::generate_noise_signal<ValueType>(size, std::random_device{}());
    auto const z_buf = neo::generate_noise_signal<ValueType>(size, std::random_device{}());
    auto out_buf     = stdex::mdarray<ValueType, stdex::dextents<size_t, 1>>{size};

    auto const x   = x_buf.to_mdspan();
    auto const y   = y_buf.to_mdspan();
    auto const z   = z_buf.to_mdspan();
    auto const out = out_buf.to_mdspan();

    for (auto _ : state) {
        state.PauseTiming();
        neo::copy(z, out);
        state.ResumeTiming();

        neo::multiply_add(x, y, out, out);

        benchmark::DoNotOptimize(out_buf.data());
        benchmark::DoNotOptimize(out[0]);
        benchmark::ClobberMemory();
    }

    auto const items = static_cast<int64_t>(state.iterations()) * state.range(0);
    state.SetItemsProcessed(items);
    state.SetBytesProcessed(items * sizeof(ValueType));
}

}  // namespace

BENCHMARK(multiply_add<float>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);
BENCHMARK(multiply_add<double>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);

BENCHMARK(multiply_add<neo::q7>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);
BENCHMARK(multiply_add<neo::q15>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);

// BENCHMARK(multiply_add<neo::q15x8>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);
// BENCHMARK(multiply_add<neo::q15x16>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);

BENCHMARK(multiply_add<std::complex<float>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);
BENCHMARK(multiply_add<std::complex<double>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);

BENCHMARK(multiply_add<neo::complex64>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);
BENCHMARK(multiply_add<neo::complex128>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);

BENCHMARK_MAIN();
