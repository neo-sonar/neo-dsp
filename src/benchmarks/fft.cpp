#include "neo/fft.hpp"

#include <neo/fft/testing/benchmark.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <random>
#include <vector>

template<typename Float>
struct cfft
{
    explicit cfft(size_t size) : _buf(size, Float(0)), _tw{neo::fft::make_radix2_twiddles<std::complex<Float>>(size)} {}

    auto operator()() -> void
    {
        auto const gen = [i = 0]() mutable { return static_cast<Float>(i++); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::fft_radix2_kernel_v1(Kokkos::mdspan{_buf.data(), Kokkos::extents{_buf.size()}}, _tw);
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::vector<std::complex<Float>> _buf;
    std::vector<std::complex<Float>> _tw;
};

template<typename Float, unsigned Size>
struct cfft_fixed
{
    cfft_fixed() = default;

    auto operator()() -> void
    {
        auto buffer = _buffer.to_mdspan();
        // auto const gen = [i = 0]() mutable { return static_cast<Float>(i++); };
        // std::generate_n(begin(_buffer), size(_buffer), gen);
        neo::fft::fft_radix2_kernel_v1(buffer, _tw);
        neo::fft::do_not_optimize(buffer[0]);
    }

private:
    KokkosEx::mdarray<std::complex<Float>, Kokkos::extents<size_t, Size>> _buffer{};
    std::array<std::complex<Float>, Size / 2> _tw{neo::fft::make_radix2_twiddles<std::complex<Float>, Size>()};
};

template<typename Float>
struct cfft_alt
{
    explicit cfft_alt(size_t size)
        : _buf(size, Float(0))
        , _tw{neo::fft::make_radix2_twiddles<std::complex<Float>>(size)}
    {}

    auto operator()() -> void
    {
        auto const gen = [i = 0]() mutable { return static_cast<Float>(i++); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::fft_radix2_kernel_v2(Kokkos::mdspan{_buf.data(), Kokkos::extents{_buf.size()}}, _tw);
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::vector<std::complex<Float>> _buf;
    std::vector<std::complex<Float>> _tw;
};

template<typename Float>
struct cfft_plan
{
    explicit cfft_plan(size_t size) : _buf(size, Float(0)), _fft{neo::fft::ilog2(size)} {}

    auto operator()() -> void
    {
        auto const gen = [i = 0]() mutable { return static_cast<Float>(i++); };
        auto const buf = Kokkos::mdspan{_buf.data(), Kokkos::extents{_buf.size()}};
        std::generate_n(begin(_buf), size(_buf), gen);
        _fft(buf, neo::fft::direction::forward);
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::vector<std::complex<Float>> _buf;
    neo::fft::fft_radix2_plan<std::complex<Float>> _fft;
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
        neo::fft::fft_radix2_kernel_v1(Kokkos::mdspan{_buf.data(), Kokkos::extents{_buf.size()}}, _tw);
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
        neo::fft::fft_radix2_kernel_v1(Kokkos::mdspan{_buf.data(), Kokkos::extents{_buf.size()}}, _tw);
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
        neo::fft::fft_radix2_kernel_v1(Kokkos::mdspan{_buf.data(), Kokkos::extents{_buf.size()}}, _tw);
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
        neo::fft::fft_radix2_kernel_v1(
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
        neo::fft::fft_radix2_kernel_v1(Kokkos::mdspan{_buf.data(), Kokkos::extents{_buf.size()}}, _tw);
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
        neo::fft::fft_radix2_kernel_v1(Kokkos::mdspan{_buf.data(), Kokkos::extents{_buf.size()}}, _tw);
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
        neo::fft::fft_radix2_kernel_v1(Kokkos::mdspan{_buf.data(), Kokkos::extents{_buf.size()}}, _tw);
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::vector<neo::fft::complex64x4_t> _buf;
    std::vector<neo::fft::complex64x4_t> _tw;
};

#endif

auto main() -> int
{
    static constexpr auto N = 1024;

    neo::fft::benchmark_fft("cfft<complex<float>>(N)", N, 1, cfft<float>{N});
    neo::fft::benchmark_fft("cfft_alt<complex<float>>(N)", N, 1, cfft_alt<float>{N});
    neo::fft::benchmark_fft("cfft_plan<complex<float>>(N)", N, 1, cfft_plan<float>{N});
    neo::fft::benchmark_fft("cfft_fixed<complex<float>, N>()", N, 1, cfft_fixed<float, N>{});
    std::printf("\n");

    neo::fft::benchmark_fft("cfft<complex<double>>(N)", N, 1, cfft<double>{N});
    neo::fft::benchmark_fft("cfft_alt<complex<double>>(N)", N, 1, cfft_alt<double>{N});
    neo::fft::benchmark_fft("cfft_plan<complex<double>>(N)", N, 1, cfft_plan<double>{N});
    neo::fft::benchmark_fft("cfft_fixed<complex<double>, N>()", N, 1, cfft_fixed<double, N>{});
    std::printf("\n");

    // benchmark_fft("radix2<complex<float>>(N)", 2048, 1, cfft<float>{2048});
    // benchmark_fft("radix2<complex<float>>(N)", 4096, 1, cfft<float>{4096});

    // // benchmark_fft("radix2<complex<float>, N>()", 2048, 1, cfft_fixed<float, 2048>{});
    // // benchmark_fft("radix2<complex<float>, N>()", 4096, 1, cfft_fixed<float, 4096>{});

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
