#include <neo/fft.hpp>

#include <neo/testing/testing.hpp>

#include <benchmark/benchmark.h>

namespace {

template<typename Plan>
auto bench_c2c(benchmark::State& state) -> void
{
    using Complex = typename Plan::value_type;
    using Real    = typename Complex::value_type;

    auto const len   = static_cast<std::size_t>(state.range(0));
    auto const order = neo::ilog2(len);
    auto const noise = neo::generate_noise_signal<Complex>(len, std::random_device{}());

    auto plan = Plan{order};
    auto work = noise;

    for (auto _ : state) {
        state.PauseTiming();
        neo::copy(noise.to_mdspan(), work.to_mdspan());
        state.ResumeTiming();

        neo::fft::fft(plan, work.to_mdspan());

        benchmark::DoNotOptimize(work.data());
        benchmark::ClobberMemory();
    }

    auto const items = static_cast<int64_t>(state.iterations()) * state.range(0);
    state.SetItemsProcessed(items);
    state.SetBytesProcessed(items * sizeof(Complex));
}

}  // namespace

BENCHMARK(bench_c2c<neo::fft::fft_plan<std::complex<float>>>)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);
BENCHMARK(bench_c2c<neo::fft::fft_plan<std::complex<double>>>)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);

#if defined(NEO_PLATFORM_APPLE)

BENCHMARK(bench_c2c<neo::fft::apple_vdsp_fft_plan<std::complex<float>>>)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);
BENCHMARK(bench_c2c<neo::fft::apple_vdsp_fft_plan<std::complex<double>>>)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);

#endif
BENCHMARK_MAIN();
