#include <neo/fft.hpp>
#include <neo/math/simd.hpp>

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
        neo::do_not_optimize(buf[0]);
    }

private:
    stdex::mdarray<std::complex<Float>, stdex::dextents<size_t, 1>> _buf;
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
        neo::fft::execute_radix2_kernel(Kernel{}, buffer, _tw.to_mdspan());
        neo::do_not_optimize(buffer[0]);
    }

private:
    stdex::mdarray<std::complex<Float>, stdex::extents<size_t, Size>> _buffer{};
    stdex::mdarray<std::complex<Float>, stdex::extents<size_t, Size / 2>> _tw{
        neo::fft::make_radix2_twiddles<std::complex<Float>, Size>()};
};

#if defined(NEO_HAS_SIMD_SSE2)

struct cfft32x2
{
    explicit cfft32x2(size_t size) : _buf(size, _mm_set1_ps(0)), _tw(size, _mm_set1_ps(0))
    // , _tw{neo::fft::make_radix2_twiddles<std::complex<float>>(size)}
    {}

    auto operator()() -> void
    {
        auto gen = [i = 0]() mutable { return _mm_set1_ps(static_cast<float>(i++)); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::execute_radix2_kernel(
            neo::fft::radix2_kernel_v1{},
            stdex::mdspan{_buf.data(), stdex::extents{_buf.size()}},
            _tw
        );
        neo::do_not_optimize(_buf.back());
    }

private:
    std::vector<neo::simd::icomplex64x2> _buf;
    std::vector<neo::simd::icomplex64x2> _tw;
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
        neo::fft::execute_radix2_kernel(
            neo::fft::radix2_kernel_v1{},
            stdex::mdspan{_buf.data(), stdex::extents{_buf.size()}},
            _tw
        );
        neo::do_not_optimize(_buf.back());
    }

private:
    std::vector<neo::simd::icomplex128x1> _buf;
    std::vector<neo::simd::icomplex128x1> _tw;
};

#endif

#if defined(NEO_HAS_SIMD_AVX)

struct cfft32x4
{
    explicit cfft32x4(size_t size) : _buf(size, _mm256_set1_ps(0)), _tw(size, _mm256_set1_ps(0))
    // , _tw{neo::fft::make_radix2_twiddles<std::complex<float>>(size)}
    {}

    auto operator()() -> void
    {
        auto gen = [i = 0]() mutable { return _mm256_set1_ps(static_cast<float>(i++)); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::execute_radix2_kernel(
            neo::fft::radix2_kernel_v1{},
            stdex::mdspan{_buf.data(), stdex::extents{_buf.size()}},
            _tw
        );
        neo::do_not_optimize(_buf.back());
    }

private:
    std::vector<neo::simd::icomplex64x4> _buf;
    std::vector<neo::simd::icomplex64x4> _tw;
};

template<size_t Size>
struct cfft32x4_fixed
{
    cfft32x4_fixed() = default;

    auto operator()() -> void
    {
        auto gen = [i = 0]() mutable { return _mm256_set1_ps(static_cast<float>(i++)); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::execute_radix2_kernel(
            neo::fft::radix2_kernel_v1{},
            stdex::mdspan<neo::simd::icomplex64x4, stdex::extents<size_t, Size>>{_buf.data()},
            _tw
        );
        neo::do_not_optimize(_buf.back());
    }

private:
    std::array<neo::simd::icomplex64x4, Size> _buf;
    std::array<neo::simd::icomplex64x4, Size / 2> _tw;
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
        neo::fft::execute_radix2_kernel(
            neo::fft::radix2_kernel_v1{},
            stdex::mdspan{_buf.data(), stdex::extents{_buf.size()}},
            _tw
        );
        neo::do_not_optimize(_buf.back());
    }

private:
    std::vector<neo::simd::icomplex128x2> _buf;
    std::vector<neo::simd::icomplex128x2> _tw;
};
#endif

#if defined(NEO_HAS_SIMD_AVX512F)

struct cfft32x8
{
    explicit cfft32x8(size_t size) : _buf(size, _mm512_set1_ps(0)), _tw(size, _mm512_set1_ps(0))
    // , _tw{neo::fft::make_radix2_twiddles<std::complex<float>>(size)}
    {}

    auto operator()() -> void
    {
        auto gen = [i = 0]() mutable { return _mm512_set1_ps(static_cast<float>(i++)); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::execute_radix2_kernel(
            neo::fft::radix2_kernel_v1{},
            stdex::mdspan{_buf.data(), stdex::extents{_buf.size()}},
            _tw
        );
        neo::do_not_optimize(_buf.back());
    }

private:
    std::vector<neo::simd::icomplex64x8> _buf;
    std::vector<neo::simd::icomplex64x8> _tw;
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
        neo::fft::execute_radix2_kernel(
            neo::fft::radix2_kernel_v1{},
            stdex::mdspan{_buf.data(), stdex::extents{_buf.size()}},
            _tw
        );
        neo::do_not_optimize(_buf.back());
    }

private:
    std::vector<neo::simd::icomplex128x4> _buf;
    std::vector<neo::simd::icomplex128x4> _tw;
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

    neo::benchmark_fft("fft_static<complex<float>, N, v1>()", N, 1, fft_static<float, neo::fft::radix2_kernel_v1, N>{});
    neo::benchmark_fft("fft_static<complex<float>, N, v2>()", N, 1, fft_static<float, neo::fft::radix2_kernel_v2, N>{});
    neo::benchmark_fft("fft_static<complex<float>, N, v3>()", N, 1, fft_static<float, neo::fft::radix2_kernel_v3, N>{});
    neo::benchmark_fft("fft_static<complex<float>, N, v4>()", N, 1, fft_static<float, neo::fft::radix2_kernel_v4, N>{});
    std::printf("\n");

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

    // #if defined(NEO_HAS_SIMD_AVX)
    //     benchmark_fft("radix2<icomplex64x4>(N)", 2048, 4, cfft32x4{2048});
    //     benchmark_fft("radix2<icomplex64x4>(N)", 4096, 4, cfft32x4{4096});
    //     benchmark_fft("radix2<icomplex64x4>(N)", 8192, 4, cfft32x4{8192});
    //     benchmark_fft("radix2<icomplex64x4, N>()", 2048, 4, cfft32x4_fixed<2048>{});
    //     benchmark_fft("radix2<icomplex64x4, N>()", 4096, 4, cfft32x4_fixed<4096>{});
    //     benchmark_fft("radix2<icomplex64x4, N>()", 8192, 4, cfft32x4_fixed<8192>{});
    // #endif

    return EXIT_SUCCESS;
}
