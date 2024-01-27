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

    auto const block_size   = static_cast<std::size_t>(state.range(0));
    auto const impulse_size = static_cast<std::size_t>(state.range(1));

    auto const impulse = [impulse_size] {
        auto buf = neo::generate_noise_signal<Real>(impulse_size, std::random_device{}());
        neo::convolution::normalize_impulse(buf.to_mdspan());
        return buf;
    }();
    auto const matrix = stdex::mdspan{impulse.data(), stdex::extents(1, impulse.extent(0))};
    auto const filter = neo::convolution::uniform_partition(matrix, block_size);

    auto convolver = Convolver{};
    convolver.filter(stdex::submdspan(filter.to_mdspan(), 0, stdex::full_extent, stdex::full_extent));

    auto const noise = neo::generate_noise_signal<Real>(block_size, std::random_device{}());
    auto block       = noise;

    for (auto _ : state) {
        neo::copy(noise.to_mdspan(), block.to_mdspan());
        convolver(block.to_mdspan());

        benchmark::DoNotOptimize(block(0));
        benchmark::ClobberMemory();
    }

    auto const items = static_cast<int64_t>(state.iterations()) * block_size;
    state.SetItemsProcessed(items);
    state.SetBytesProcessed(items * sizeof(Real));
}

constexpr auto const min_block  = 512;
constexpr auto const max_block  = 512;
constexpr auto const min_filter = 1 << 11;
constexpr auto const max_filter = 1 << 17;

}  // namespace

BENCHMARK(conv<neo::convolution::upols_convolver<std::complex<float>>>)
    ->ArgsProduct({benchmark::CreateRange(min_block, max_block, 2), benchmark::CreateRange(min_filter, max_filter, 2)});
BENCHMARK(conv<neo::convolution::upola_convolver<std::complex<float>>>)
    ->ArgsProduct({benchmark::CreateRange(min_block, max_block, 2), benchmark::CreateRange(min_filter, max_filter, 2)});
BENCHMARK(conv<neo::convolution::upola_convolver_v2<std::complex<float>>>)
    ->ArgsProduct({benchmark::CreateRange(min_block, max_block, 2), benchmark::CreateRange(min_filter, max_filter, 2)});

BENCHMARK(conv<neo::convolution::split_upola_convolver<std::complex<float>>>)
    ->ArgsProduct({benchmark::CreateRange(min_block, max_block, 2), benchmark::CreateRange(min_filter, max_filter, 2)});
BENCHMARK(conv<neo::convolution::split_upols_convolver<std::complex<float>>>)
    ->ArgsProduct({benchmark::CreateRange(min_block, max_block, 2), benchmark::CreateRange(min_filter, max_filter, 2)});

BENCHMARK_MAIN();
