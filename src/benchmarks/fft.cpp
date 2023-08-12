#include "neo/fft.hpp"

#include <neo/testing/benchmark.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <random>
#include <vector>

template<typename Float, typename Kernel>
struct fft_plan
{
    explicit fft_plan(size_t size) : _buf(size), _fft{neo::ilog2(size)} {}

    auto operator()() -> void
    {
        auto const gen = [i = 0]() mutable { return static_cast<Float>(i++); };
        auto const buf = _buf.to_mdspan();
        std::generate_n(_buf.data(), _buf.size(), gen);
        _fft(buf, neo::fft::direction::forward);
        neo::fft::do_not_optimize(buf[0]);
    }

private:
    KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>> _buf;
    neo::fft::fft_radix2_plan<std::complex<Float>, Kernel> _fft;
};

template<typename Float, typename Kernel, unsigned Size>
struct fft_static
{
    fft_static() = default;

    auto operator()() -> void
    {
        auto const buffer = _buffer.to_mdspan();
        auto const gen    = [i = 0]() mutable { return static_cast<Float>(i++); };
        std::generate_n(buffer.data_handle(), buffer.size(), gen);
        neo::fft::fft_radix2(Kernel{}, buffer, _tw);
        neo::fft::do_not_optimize(buffer[0]);
    }

private:
    KokkosEx::mdarray<std::complex<Float>, Kokkos::extents<size_t, Size>> _buffer{};
    KokkosEx::mdarray<std::complex<Float>, Kokkos::extents<size_t, Size / 2>> _tw{
        neo::fft::make_radix2_twiddles<std::complex<Float>, Size>()};
};

#if defined(__amd64__) or defined(_M_AMD64)
struct cfft32x2
{
    explicit cfft32x2(size_t size) : _buf(size, _mm_set1_ps(0)), _tw(size, _mm_set1_ps(0))
    // , _tw{neo::fft::make_radix2_twiddles<std::complex<float>>(size)}
    {}

    auto operator()() -> void
    {
        auto gen = [i = 0]() mutable { return _mm_set1_ps(static_cast<float>(i++)); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::fft_radix2(
            neo::fft::radix2_kernel_v1{},
            Kokkos::mdspan{_buf.data(), Kokkos::extents{_buf.size()}},
            _tw
        );
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::vector<neo::fft::complex32x2_t> _buf;
    std::vector<neo::fft::complex32x2_t> _tw;
};

struct cfft64x1
{
    explicit cfft64x1(size_t size) : _buf(size, _mm_set1_pd(0)), _tw(size, _mm_set1_pd(0))
    // , _tw{neo::fft::make_radix2_twiddles<std::complex<float>>(size)}
    {}

    auto operator()() -> void
    {
        auto gen = [i = 0]() mutable { return _mm_set1_pd(static_cast<double>(i++)); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::fft_radix2(
            neo::fft::radix2_kernel_v1{},
            Kokkos::mdspan{_buf.data(), Kokkos::extents{_buf.size()}},
            _tw
        );
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::vector<neo::fft::complex64x1_t> _buf;
    std::vector<neo::fft::complex64x1_t> _tw;
};

#endif

#ifdef __AVX__

struct cfft32x4
{
    explicit cfft32x4(size_t size) : _buf(size, _mm256_set1_ps(0)), _tw(size, _mm256_set1_ps(0))
    // , _tw{neo::fft::make_radix2_twiddles<std::complex<float>>(size)}
    {}

    auto operator()() -> void
    {
        auto gen = [i = 0]() mutable { return _mm256_set1_ps(static_cast<float>(i++)); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::fft_radix2(
            neo::fft::radix2_kernel_v1{},
            Kokkos::mdspan{_buf.data(), Kokkos::extents{_buf.size()}},
            _tw
        );
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::vector<neo::fft::complex32x4_t> _buf;
    std::vector<neo::fft::complex32x4_t> _tw;
};

template<size_t Size>
struct cfft32x4_fixed
{
    cfft32x4_fixed() = default;

    auto operator()() -> void
    {
        auto gen = [i = 0]() mutable { return _mm256_set1_ps(static_cast<float>(i++)); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::fft_radix2(
            neo::fft::radix2_kernel_v1{},
            Kokkos::mdspan<neo::fft::complex32x4_t, Kokkos::extents<size_t, Size>>{_buf.data()},
            _tw
        );
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::array<neo::fft::complex32x4_t, Size> _buf;
    std::array<neo::fft::complex32x4_t, Size / 2> _tw;
};

struct cfft64x2
{
    explicit cfft64x2(size_t size) : _buf(size, _mm256_set1_pd(0)), _tw(size, _mm256_set1_pd(0))
    // , _tw{neo::fft::make_radix2_twiddles<std::complex<float>>(size)}
    {}

    auto operator()() -> void
    {
        auto gen = [i = 0]() mutable { return _mm256_set1_pd(static_cast<double>(i++)); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::fft_radix2(
            neo::fft::radix2_kernel_v1{},
            Kokkos::mdspan{_buf.data(), Kokkos::extents{_buf.size()}},
            _tw
        );
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::vector<neo::fft::complex64x2_t> _buf;
    std::vector<neo::fft::complex64x2_t> _tw;
};
#endif

#ifdef __AVX512F__

struct cfft32x8
{
    explicit cfft32x8(size_t size) : _buf(size, _mm512_set1_ps(0)), _tw(size, _mm512_set1_ps(0))
    // , _tw{neo::fft::make_radix2_twiddles<std::complex<float>>(size)}
    {}

    auto operator()() -> void
    {
        auto gen = [i = 0]() mutable { return _mm512_set1_ps(static_cast<float>(i++)); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::fft_radix2(
            neo::fft::radix2_kernel_v1{},
            Kokkos::mdspan{_buf.data(), Kokkos::extents{_buf.size()}},
            _tw
        );
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::vector<neo::fft::complex32x8_t> _buf;
    std::vector<neo::fft::complex32x8_t> _tw;
};

struct cfft64x4
{
    explicit cfft64x4(size_t size) : _buf(size, _mm512_set1_pd(0)), _tw(size, _mm512_set1_pd(0))
    // , _tw{neo::fft::make_radix2_twiddles<std::complex<float>>(size)}
    {}

    auto operator()() -> void
    {
        auto gen = [i = 0]() mutable { return _mm512_set1_pd(static_cast<double>(i++)); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::fft_radix2(
            neo::fft::radix2_kernel_v1{},
            Kokkos::mdspan{_buf.data(), Kokkos::extents{_buf.size()}},
            _tw
        );
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::vector<neo::fft::complex64x4_t> _buf;
    std::vector<neo::fft::complex64x4_t> _tw;
};

#endif

auto main() -> int
{
    namespace fft           = neo::fft;
    static constexpr auto N = 1024;

    neo::benchmark_fft("fft_plan<complex<float>, v1>(N)", N, 1, fft_plan<float, neo::fft::radix2_kernel_v1>{N});
    neo::benchmark_fft("fft_plan<complex<float>, v2>(N)", N, 1, fft_plan<float, neo::fft::radix2_kernel_v2>{N});
    neo::benchmark_fft("fft_plan<complex<float>, v3>(N)", N, 1, fft_plan<float, neo::fft::radix2_kernel_v3>{N});
    neo::benchmark_fft("fft_plan<complex<float>, v4>(N)", N, 1, fft_plan<float, neo::fft::radix2_kernel_v4>{N});
    // std::printf("\n");

    // neo::benchmark_fft("fft_static<complex<float>, N, v1>()", N, 1, fft_static<float, neo::fft::radix2_kernel_v1,
    // N>{}); neo::benchmark_fft("fft_static<complex<float>, N, v2>()", N, 1, fft_static<float,
    // neo::fft::radix2_kernel_v2, N>{}); neo::benchmark_fft("fft_static<complex<float>, N, v3>()", N, 1,
    // fft_static<float, neo::fft::radix2_kernel_v3, N>{}); std::printf("\n");

    // neo::benchmark_fft("fft_plan<complex<double>, v1>(N)", N, 1, fft_plan<double, neo::fft::radix2_kernel_v1>{N});
    // neo::benchmark_fft("fft_plan<complex<double>, v2>(N)", N, 1, fft_plan<double, neo::fft::radix2_kernel_v2>{N});
    // neo::benchmark_fft("fft_plan<complex<double>, v3>(N)", N, 1, fft_plan<double, neo::fft::radix2_kernel_v3>{N});
    // std::printf("\n");

    // neo::benchmark_fft("fft_static<complex<double>, N, v1>()", N, 1, fft_static<double, neo::fft::radix2_kernel_v1,
    // N>{}); neo::benchmark_fft("fft_static<complex<double>, N, v2>()", N, 1, fft_static<double,
    // neo::fft::radix2_kernel_v2, N>{}); neo::benchmark_fft("fft_static<complex<double>, N, v3>()", N, 1,
    // fft_static<double, neo::fft::radix2_kernel_v3, N>{}); std::printf("\n");

    // benchmark_fft("radix2<complex<float>>(N)", 2048, 1, cfft<float>{2048});
    // benchmark_fft("radix2<complex<float>>(N)", 4096, 1, cfft<float>{4096});

    // // benchmark_fft("radix2<complex<float>, N>()", 2048, 1, fft_static<float, 2048>{});
    // // benchmark_fft("radix2<complex<float>, N>()", 4096, 1, fft_static<float, 4096>{});

    // #ifdef __AVX__
    //     benchmark_fft("radix2<complex32x4>(N)", 2048, 4, cfft32x4{2048});
    //     benchmark_fft("radix2<complex32x4>(N)", 4096, 4, cfft32x4{4096});
    //     benchmark_fft("radix2<complex32x4>(N)", 8192, 4, cfft32x4{8192});
    //     benchmark_fft("radix2<complex32x4, N>()", 2048, 4, cfft32x4_fixed<2048>{});
    //     benchmark_fft("radix2<complex32x4, N>()", 4096, 4, cfft32x4_fixed<4096>{});
    //     benchmark_fft("radix2<complex32x4, N>()", 8192, 4, cfft32x4_fixed<8192>{});
    // #endif

    return EXIT_SUCCESS;
}
