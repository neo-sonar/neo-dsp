#include <neo/algorithm.hpp>
#include <neo/complex.hpp>
#include <neo/fft.hpp>

#include <neo/testing/benchmark.hpp>
#include <neo/testing/testing.hpp>

#include <fmt/format.h>
#include <fmt/os.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string_view>
#include <utility>
#include <vector>

namespace {

template<typename Func>
auto timeit(std::string_view name, size_t size_of_t, size_t n, Func func)
{
    using microseconds = std::chrono::duration<double, std::micro>;

    auto const size       = n;
    auto const iterations = 25'000U;
    auto const margin     = iterations / 20U;

    auto all_runs = std::vector<double>(iterations);

    func();
    func();
    func();

    for (auto i{0U}; i < iterations; ++i) {
        auto start = std::chrono::system_clock::now();
        func();
        auto stop = std::chrono::system_clock::now();

        all_runs[i] = std::chrono::duration_cast<microseconds>(stop - start).count();
    }

    auto const runs = std::span{all_runs}.subspan(margin, all_runs.size() - static_cast<std::size_t>(margin) * 2);
    auto const avg  = std::reduce(runs.begin(), runs.end(), 0.0) / double(runs.size());
    auto const items_per_sec      = static_cast<int>(std::lround(double(size) / avg));
    auto const mega_bytes_per_sec = std::round(double(size * size_of_t) / avg) / 1000.0;
    fmt::println(
        "{:<32} avg: {:.1f}us - GB/sec: {:.2f} - N/usec: {}",
        name.data(),
        avg,
        mega_bytes_per_sec,
        items_per_sec
    );
}

template<typename Float>
struct split_complex_fma
{
    explicit split_complex_fma(size_t size) : _x{2, size}, _y{2, size}, _out{2, size}
    {
        auto const noise_x = neo::generate_noise_signal<neo::scalar_complex<Float>>(size, std::random_device{}());
        auto const noise_y = neo::generate_noise_signal<neo::scalar_complex<Float>>(size, std::random_device{}());
        for (auto i{0U}; i < size; ++i) {
            _x(0, i) = noise_x(i).real();
            _x(1, i) = noise_x(i).imag();
            _y(0, i) = noise_y(i).real();
            _y(1, i) = noise_y(i).imag();
        }
    }

    auto operator()() noexcept -> void
    {
        auto const x = neo::split_complex{
            stdex::submdspan(_x.to_mdspan(), 0, stdex::full_extent),
            stdex::submdspan(_x.to_mdspan(), 1, stdex::full_extent),
        };
        auto const y = neo::split_complex{
            stdex::submdspan(_y.to_mdspan(), 0, stdex::full_extent),
            stdex::submdspan(_y.to_mdspan(), 1, stdex::full_extent),
        };
        auto const out = neo::split_complex{
            stdex::submdspan(_out.to_mdspan(), 0, stdex::full_extent),
            stdex::submdspan(_out.to_mdspan(), 1, stdex::full_extent),
        };

        neo::multiply_add(x, y, out, out);

        neo::do_not_optimize(out.real[0]);
        neo::do_not_optimize(out.imag[0]);
    }

private:
    stdex::mdarray<Float, stdex::dextents<size_t, 2>> _x;
    stdex::mdarray<Float, stdex::dextents<size_t, 2>> _y;
    stdex::mdarray<Float, stdex::dextents<size_t, 2>> _out;
};

#if defined(NEO_HAS_SIMD_AVX)
template<typename Float>
struct split_complex_fma_avx
{
    explicit split_complex_fma_avx(size_t size) : _x{2, size}, _y{2, size}, _out{2, size}
    {
        auto const noise_x = neo::generate_noise_signal<neo::scalar_complex<Float>>(size, std::random_device{}());
        auto const noise_y = neo::generate_noise_signal<neo::scalar_complex<Float>>(size, std::random_device{}());
        for (auto i{0U}; i < size; ++i) {
            _x(0, i) = noise_x(i).real();
            _x(1, i) = noise_x(i).imag();
            _y(0, i) = noise_y(i).real();
            _y(1, i) = noise_y(i).imag();
        }
    }

    auto operator()() noexcept -> void
    {
        auto const x_vec = neo::split_complex{
            stdex::submdspan(_x.to_mdspan(), 0, stdex::full_extent),
            stdex::submdspan(_x.to_mdspan(), 1, stdex::full_extent),
        };
        auto const y_vec = neo::split_complex{
            stdex::submdspan(_y.to_mdspan(), 0, stdex::full_extent),
            stdex::submdspan(_y.to_mdspan(), 1, stdex::full_extent),
        };
        auto const out_vec = neo::split_complex{
            stdex::submdspan(_out.to_mdspan(), 0, stdex::full_extent),
            stdex::submdspan(_out.to_mdspan(), 1, stdex::full_extent),
        };

        auto const register_size = 256 / 8 / sizeof(Float);
        for (auto i{0}; i < static_cast<int>(_x.extent(1)); i += register_size) {
            auto const xre = _mm256_loadu_ps(std::addressof(x_vec.real[i]));
            auto const xim = _mm256_loadu_ps(std::addressof(x_vec.imag[i]));

            auto const yre = _mm256_loadu_ps(std::addressof(y_vec.real[i]));
            auto const yim = _mm256_loadu_ps(std::addressof(y_vec.imag[i]));

            auto const zre = _mm256_loadu_ps(std::addressof(out_vec.real[i]));
            auto const zim = _mm256_loadu_ps(std::addressof(out_vec.imag[i]));

            auto const product_re = _mm256_sub_ps(_mm256_mul_ps(xre, yre), _mm256_mul_ps(xim, yim));
            auto const product_im = _mm256_add_ps(_mm256_mul_ps(xre, yim), _mm256_mul_ps(xim, yre));

            auto const sum_re = _mm256_add_ps(product_re, zre);
            auto const sum_im = _mm256_add_ps(product_im, zim);

            _mm256_storeu_ps(std::addressof(out_vec.real[i]), sum_re);
            _mm256_storeu_ps(std::addressof(out_vec.imag[i]), sum_im);
        }

        neo::do_not_optimize(out_vec.real[0]);
        neo::do_not_optimize(out_vec.imag[0]);
    }

private:
    stdex::mdarray<Float, stdex::dextents<size_t, 2>> _x;
    stdex::mdarray<Float, stdex::dextents<size_t, 2>> _y;
    stdex::mdarray<Float, stdex::dextents<size_t, 2>> _out;
};
#endif

}  // namespace

auto main() -> int
{
    static constexpr auto n = 131072U;

#if defined(NEO_HAS_SIMD_F16C) or defined(NEO_HAS_SIMD_NEON)
    // timeit("multiply_add(split_complex<_Float16>):    ", 4, n, split_complex_fma<_Float16>{n});
#endif
    timeit("multiply_add(split_complex<float>):    ", 4, n, split_complex_fma<float>{n});
    timeit("multiply_add(split_complex<double>):   ", 8, n, split_complex_fma<double>{n});
    std::puts("\n");

#if defined(NEO_HAS_SIMD_AVX)
    timeit("multiply_add(split_complex<float>):  ", 4, n, split_complex_fma_avx<float>{n});
    // timeit("multiply_add(split_complex<double>): ", 8, n, split_complex_fma_avx<double>{n});
#endif
    return EXIT_SUCCESS;
}
