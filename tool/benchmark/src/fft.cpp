#include <neo/fft.hpp>

#include <neo/testing/testing.hpp>

#include <benchmark/benchmark.h>

namespace {

template<typename Plan>
auto c2c(benchmark::State& state) -> void
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

    auto const flop        = 5UL * plan.size() * plan.order() * static_cast<size_t>(state.iterations());
    state.counters["flop"] = benchmark::Counter(static_cast<double>(flop), benchmark::Counter::kIsRate);
}

template<typename Plan>
auto split_c2c(benchmark::State& state) -> void
{
    using Float = typename Plan::value_type;

    auto const len   = static_cast<std::size_t>(state.range(0));
    auto const order = neo::ilog2(len);
    auto const noise = neo::generate_noise_signal<Float>(len, std::random_device{}());

    auto plan = Plan{order};
    auto buf  = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{2, len};
    auto z    = neo::split_complex{
        stdex::submdspan(buf.to_mdspan(), 0, stdex::full_extent),
        stdex::submdspan(buf.to_mdspan(), 1, stdex::full_extent),
    };

    for (auto _ : state) {
        state.PauseTiming();
        neo::copy(noise.to_mdspan(), z.real);
        neo::fill(z.imag, Float(0));
        state.ResumeTiming();

        neo::fft::fft(plan, z);

        benchmark::DoNotOptimize(z.real[0]);
        benchmark::DoNotOptimize(z.imag[0]);
        benchmark::ClobberMemory();
    }

    auto const flop        = 5UL * plan.size() * plan.order() * static_cast<size_t>(state.iterations());
    state.counters["flop"] = benchmark::Counter(static_cast<double>(flop), benchmark::Counter::kIsRate);
}

}  // namespace

#if defined(NEO_HAS_INTEL_IPP)
BENCHMARK(split_c2c<neo::fft::intel_ipp_split_fft_plan<float>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 13);
BENCHMARK(split_c2c<neo::fft::intel_ipp_split_fft_plan<double>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 13);
#endif

#if defined(NEO_HAS_APPLE_VDSP)
BENCHMARK(c2c<neo::fft::apple_vdsp_fft_plan<neo::complex64>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 13);
BENCHMARK(c2c<neo::fft::apple_vdsp_fft_plan<neo::complex128>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 13);
#endif

#if defined(NEO_HAS_INTEL_IPP)
BENCHMARK(c2c<neo::fft::intel_ipp_fft_plan<neo::complex64>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 13);
BENCHMARK(c2c<neo::fft::intel_ipp_fft_plan<neo::complex128>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 13);
#endif

#if defined(NEO_HAS_INTEL_MKL)
BENCHMARK(c2c<neo::fft::intel_mkl_fft_plan<neo::complex64>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 13);
BENCHMARK(c2c<neo::fft::intel_mkl_fft_plan<neo::complex128>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 13);
#endif

BENCHMARK(c2c<neo::fft::fallback_fft_plan<neo::complex64>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 13);
BENCHMARK(c2c<neo::fft::fallback_fft_plan<neo::complex128>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 13);

BENCHMARK(c2c<neo::fft::fft_plan<neo::complex64>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 13);
BENCHMARK(c2c<neo::fft::fft_plan<neo::complex128>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 13);

BENCHMARK_MAIN();
