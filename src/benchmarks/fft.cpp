#include <neo/algorithm.hpp>
#include <neo/fft.hpp>
#include <neo/simd.hpp>

#include <neo/testing/benchmark.hpp>
#include <neo/testing/testing.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <string_view>

namespace {

template<typename Func>
auto timeit(std::string_view name, size_t N, Func func)
{
    using microseconds = std::chrono::duration<double, std::micro>;

    auto const num_runs = 50'000ULL;
    auto const padding  = num_runs / 100ULL * 2ULL;

    auto all_runs = stdex::mdarray<double, stdex::dextents<size_t, 1>>{num_runs};

    for (auto i{0}; i < num_runs; ++i) {
        auto start = std::chrono::system_clock::now();
        func();
        auto stop = std::chrono::system_clock::now();

        all_runs(i) = std::chrono::duration_cast<microseconds>(stop - start).count();
    }

    std::sort(all_runs.data(), std::next(all_runs.data(), all_runs.size()));

    auto const runs   = stdex::submdspan(all_runs.to_mdspan(), std::tuple{padding, all_runs.extent(0) - padding});
    auto const dsize  = static_cast<double>(N);
    auto const avg    = neo::mean(runs).value();
    auto const stddev = neo::standard_deviation(runs).value();
    auto const mflops = static_cast<int>(std::lround(5.0 * dsize * std::log2(dsize) / avg)) * 2;

    std::printf(
        "%-32s N: %-5zu - runs: %zu - avg: %.1fus - stddev: %.1fus - min: %.1fus - max: %.1fus - mflops: %d\n",
        name.data(),
        N,
        all_runs.extent(0),
        avg,
        stddev,
        runs[0],
        runs[runs.extent(0) - 1U],
        mflops
    );
}

template<typename Complex, typename Kernel>
struct fft_roundtrip
{
    explicit fft_roundtrip(size_t size)
        : _plan{neo::ilog2(size)}
        , _buffer(neo::generate_noise_signal<Complex>(_plan.size(), std::random_device{}()))
    {}

    auto operator()() -> void
    {
        auto const buffer = _buffer.to_mdspan();

        neo::fft::fft(_plan, buffer);
        neo::fft::ifft(_plan, buffer);

        neo::do_not_optimize(buffer[0]);
        neo::do_not_optimize(buffer[buffer.extent(0) - 1]);
    }

private:
    neo::fft::fft_radix2_plan<Complex, Kernel> _plan;
    stdex::mdarray<Complex, stdex::dextents<size_t, 1>> _buffer;
};

}  // namespace

namespace fft = neo::fft;

auto main(int argc, char** argv) -> int
{
    auto N = 1024UL;
    if (argc == 2) {
        N = std::stoul(argv[1]);
    }

    timeit("fft<std::complex<float>, v1>", N, fft_roundtrip<std::complex<float>, fft::radix2_kernel_v1>{N});
    timeit("fft<std::complex<float>, v2>", N, fft_roundtrip<std::complex<float>, fft::radix2_kernel_v2>{N});
    timeit("fft<std::complex<float>, v3>", N, fft_roundtrip<std::complex<float>, fft::radix2_kernel_v3>{N});
    timeit("fft<std::complex<float>, v4>", N, fft_roundtrip<std::complex<float>, fft::radix2_kernel_v4>{N});
    std::printf("\n");

    timeit("fft<neo::complex64, v1>", N, fft_roundtrip<neo::complex64, fft::radix2_kernel_v1>{N});
    timeit("fft<neo::complex64, v2>", N, fft_roundtrip<neo::complex64, fft::radix2_kernel_v2>{N});
    timeit("fft<neo::complex64, v3>", N, fft_roundtrip<neo::complex64, fft::radix2_kernel_v3>{N});
    timeit("fft<neo::complex64, v4>", N, fft_roundtrip<neo::complex64, fft::radix2_kernel_v4>{N});
    std::printf("\n");

    return EXIT_SUCCESS;
}
