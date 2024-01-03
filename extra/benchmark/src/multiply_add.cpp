// SPDX-License-Identifier: MIT

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

        benchmark::DoNotOptimize(out[0]);
        benchmark::ClobberMemory();
    }

    auto const items = static_cast<int64_t>(state.iterations()) * state.range(0);
    state.SetItemsProcessed(items);
    state.SetBytesProcessed(items * sizeof(ValueType));
}

template<typename Float>
auto split_multiply_add(benchmark::State& state) -> void
{
    auto const size = state.range(0);

    auto x_buf   = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{2, size};
    auto y_buf   = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{2, size};
    auto out_buf = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{2, size};

    auto const x = neo::split_complex{
        stdex::submdspan(x_buf.to_mdspan(), 0, stdex::full_extent),
        stdex::submdspan(x_buf.to_mdspan(), 1, stdex::full_extent),
    };
    auto const y = neo::split_complex{
        stdex::submdspan(y_buf.to_mdspan(), 0, stdex::full_extent),
        stdex::submdspan(y_buf.to_mdspan(), 1, stdex::full_extent),
    };
    auto const out = neo::split_complex{
        stdex::submdspan(out_buf.to_mdspan(), 0, stdex::full_extent),
        stdex::submdspan(out_buf.to_mdspan(), 1, stdex::full_extent),
    };

    auto const noise_x   = neo::generate_noise_signal<neo::scalar_complex<Float>>(size, std::random_device{}());
    auto const noise_y   = neo::generate_noise_signal<neo::scalar_complex<Float>>(size, std::random_device{}());
    auto const noise_out = neo::generate_noise_signal<neo::scalar_complex<Float>>(size, std::random_device{}());

    for (auto i{0U}; i < size; ++i) {
        x.real[i] = noise_x(i).real();
        x.imag[i] = noise_x(i).imag();
        y.real[i] = noise_y(i).real();
        y.imag[i] = noise_y(i).imag();
    }

    for (auto _ : state) {
        state.PauseTiming();
        for (auto i{0U}; i < size; ++i) {
            out.real[i] = noise_out(i).real();
            out.imag[i] = noise_out(i).imag();
        }
        state.ResumeTiming();

        neo::multiply_add(x, y, out, out);

        benchmark::DoNotOptimize(out.real[0]);
        benchmark::DoNotOptimize(out.imag[0]);
        benchmark::ClobberMemory();
    }

    auto const items = static_cast<int64_t>(state.iterations()) * size;
    state.SetItemsProcessed(items);
    state.SetBytesProcessed(items * sizeof(Float) * 2);
}

}  // namespace

// BENCHMARK(multiply_add<float>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);
// BENCHMARK(multiply_add<double>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);

// BENCHMARK(multiply_add<neo::q7>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);
// BENCHMARK(multiply_add<neo::q15>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);

BENCHMARK(multiply_add<std::complex<float>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);
BENCHMARK(multiply_add<std::complex<double>>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);

// BENCHMARK(multiply_add<neo::complex64>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);
// BENCHMARK(multiply_add<neo::complex128>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);

BENCHMARK(split_multiply_add<float>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);
BENCHMARK(split_multiply_add<double>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);

BENCHMARK_MAIN();
