// SPDX-License-Identifier: MIT
#include <neo/convolution.hpp>

#include <neo/testing/testing.hpp>

#include <benchmark/benchmark.h>

namespace {

template<typename Convolver>
auto conv(benchmark::State& state) -> void
{
    using Complex = typename Convolver::value_type;
    using Real    = typename Complex::value_type;

    auto const block_size = static_cast<std::size_t>(state.range(0));
    auto const len        = static_cast<std::size_t>(state.range(1));

    auto const impulse = [len] {
        auto buf = neo::generate_noise_signal<Real>(len, std::random_device{}());
        neo::normalize_impulse(buf.to_mdspan());
        return buf;
    }();
    auto const matrix = stdex::mdspan{impulse.data(), stdex::extents(1, impulse.extent(0))};
    auto const filter = neo::uniform_partition(matrix, block_size);

    auto convolver = Convolver{};
    convolver.filter(stdex::submdspan(filter.to_mdspan(), 0, stdex::full_extent, stdex::full_extent));

    auto const noise = neo::generate_noise_signal<Real>(block_size, std::random_device{}());
    auto block       = noise;

    for (auto _ : state) {
        state.PauseTiming();
        neo::copy(noise.to_mdspan(), block.to_mdspan());
        state.ResumeTiming();

        convolver(block.to_mdspan());

        benchmark::DoNotOptimize(block.data());
        benchmark::ClobberMemory();
    }

    auto const items = static_cast<int64_t>(state.iterations()) * block_size;
    state.SetItemsProcessed(items);
    state.SetBytesProcessed(items * sizeof(Real));
}

}  // namespace

BENCHMARK(conv<neo::upola_convolver<std::complex<float>>>)
    ->ArgsProduct({benchmark::CreateRange(64, 8192, 2), benchmark::CreateRange(1 << 7, 1 << 18, 2)});
BENCHMARK(conv<neo::upols_convolver<std::complex<float>>>)
    ->ArgsProduct({benchmark::CreateRange(64, 8192, 2), benchmark::CreateRange(1 << 7, 1 << 18, 2)});
// BENCHMARK(conv<neo::upola_convolver<std::complex<double>>>)
//     ->ArgsProduct({benchmark::CreateRange(64, 8192, 2), benchmark::CreateRange(1 << 7, 1 << 18, 2)});
// BENCHMARK(conv<neo::upols_convolver<std::complex<double>>>)
//     ->ArgsProduct({benchmark::CreateRange(64, 8192, 2), benchmark::CreateRange(1 << 7, 1 << 18, 2)});

BENCHMARK_MAIN();
