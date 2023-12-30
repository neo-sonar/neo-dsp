// SPDX-License-Identifier: MIT

#include <neo/fft.hpp>

#include <neo/testing/testing.hpp>

#include <benchmark/benchmark.h>

namespace {

template<typename Float>
auto multiply_add(benchmark::State& state) -> void
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

BENCHMARK(multiply_add<float>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);
BENCHMARK(multiply_add<double>)->RangeMultiplier(2)->Range(1 << 7, 1 << 20);

BENCHMARK_MAIN();
