#include "neo/fft.hpp"

#include "fft_benchmark.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <optional>
#include <random>
#include <vector>

template<typename Float>
struct cfft
{
    explicit cfft(size_t size) : _buf(size, Float(0)), _tw{neo::fft::twiddle_table_radix2<std::complex<Float>>(size)} {}

    auto operator()() -> void
    {
        auto const gen = [i = 0]() mutable { return static_cast<Float>(i++); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::c2c_radix2(std::span{_buf}, _tw);
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::vector<std::complex<Float>> _buf;
    std::vector<std::complex<Float>> _tw;
};

template<typename Float, unsigned Size>
struct cfft_fixed
{
    cfft_fixed() : _tw{neo::fft::twiddle_table_radix2<std::complex<Float>, Size>()} {}

    auto operator()() -> void
    {
        auto const gen = [i = 0]() mutable { return static_cast<Float>(i++); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::c2c_radix2<std::complex<Float>, Size>(_buf, _tw);
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::array<std::complex<Float>, Size> _buf;
    std::array<std::complex<Float>, Size / 2> _tw;
};

template<typename Float>
struct cfft_alt
{
    explicit cfft_alt(size_t size)
        : _buf(size, Float(0))
        , _tw{neo::fft::twiddle_table_radix2<std::complex<Float>>(size)}
    {}

    auto operator()() -> void
    {
        auto const gen = [i = 0]() mutable { return static_cast<Float>(i++); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::c2c_radix2_alt(std::span{_buf}, _tw);
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::vector<std::complex<Float>> _buf;
    std::vector<std::complex<Float>> _tw;
};

template<typename Float>
struct cfft_plan
{
    explicit cfft_plan(size_t size) : _buf(size, Float(0)), _fft{size} {}

    auto operator()() -> void
    {
        auto const gen = [i = 0]() mutable { return static_cast<Float>(i++); };
        std::generate_n(begin(_buf), size(_buf), gen);
        _fft(_buf, neo::fft::direction::forward);
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::vector<std::complex<Float>> _buf;
    neo::fft::c2c_radix2_plan<std::complex<Float>> _fft;
};

#if defined(__amd64__) or defined(_M_AMD64)
struct cfft32x2
{
    explicit cfft32x2(size_t size) : _buf(size, _mm_set1_ps(0)), _tw(size, _mm_set1_ps(0))
    // , _tw{neo::fft::twiddle_table_radix2<std::complex<float>>(size)}
    {}

    auto operator()() -> void
    {
        auto gen = [i = 0]() mutable { return _mm_set1_ps(static_cast<float>(i++)); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::c2c_radix2(std::span{_buf}, _tw);
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::vector<neo::fft::complex32x2_t> _buf;
    std::vector<neo::fft::complex32x2_t> _tw;
};

struct cfft64x1
{
    explicit cfft64x1(size_t size) : _buf(size, _mm_set1_pd(0)), _tw(size, _mm_set1_pd(0))
    // , _tw{neo::fft::twiddle_table_radix2<std::complex<float>>(size)}
    {}

    auto operator()() -> void
    {
        auto gen = [i = 0]() mutable { return _mm_set1_pd(static_cast<double>(i++)); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::c2c_radix2(std::span{_buf}, _tw);
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
    // , _tw{neo::fft::twiddle_table_radix2<std::complex<float>>(size)}
    {}

    auto operator()() -> void
    {
        auto gen = [i = 0]() mutable { return _mm256_set1_ps(static_cast<float>(i++)); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::c2c_radix2(std::span{_buf}, _tw);
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
        neo::fft::c2c_radix2<neo::fft::complex32x4_t, Size>(_buf, _tw);
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::array<neo::fft::complex32x4_t, Size> _buf;
    std::array<neo::fft::complex32x4_t, Size / 2> _tw;
};

struct cfft64x2
{
    explicit cfft64x2(size_t size) : _buf(size, _mm256_set1_pd(0)), _tw(size, _mm256_set1_pd(0))
    // , _tw{neo::fft::twiddle_table_radix2<std::complex<float>>(size)}
    {}

    auto operator()() -> void
    {
        auto gen = [i = 0]() mutable { return _mm256_set1_pd(static_cast<double>(i++)); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::c2c_radix2(std::span{_buf}, _tw);
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
    // , _tw{neo::fft::twiddle_table_radix2<std::complex<float>>(size)}
    {}

    auto operator()() -> void
    {
        auto gen = [i = 0]() mutable { return _mm512_set1_ps(static_cast<float>(i++)); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::c2c_radix2(std::span{_buf}, _tw);
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::vector<neo::fft::complex32x8_t> _buf;
    std::vector<neo::fft::complex32x8_t> _tw;
};

struct cfft64x4
{
    explicit cfft64x4(size_t size) : _buf(size, _mm512_set1_pd(0)), _tw(size, _mm512_set1_pd(0))
    // , _tw{neo::fft::twiddle_table_radix2<std::complex<float>>(size)}
    {}

    auto operator()() -> void
    {
        auto gen = [i = 0]() mutable { return _mm512_set1_pd(static_cast<double>(i++)); };
        std::generate_n(begin(_buf), size(_buf), gen);
        neo::fft::c2c_radix2(std::span{_buf}, _tw);
        neo::fft::do_not_optimize(_buf.back());
    }

private:
    std::vector<neo::fft::complex64x4_t> _buf;
    std::vector<neo::fft::complex64x4_t> _tw;
};

#endif

auto main() -> int
{
    using neo::fft::timeit;

    timeit("radix2<complex<float>>(N)", 2048, 1, cfft<float>{2048});
    timeit("radix2<complex<float>, N>()", 2048, 1, cfft_fixed<float, 2048>{});
#ifdef __AVX__
    timeit("radix2<complex32x4>(N)", 2048, 4, cfft32x4{2048});
    timeit("radix2<complex32x4, N>()", 2048, 4, cfft32x4_fixed<2048>{});
#endif
    std::printf("\n");

    timeit("radix2<complex<float>>(N)", 4096, 1, cfft<float>{4096});
    timeit("radix2<complex<float>, N>()", 4096, 1, cfft_fixed<float, 4096>{});
#ifdef __AVX__
    timeit("radix2<complex32x4>(N)", 4096, 4, cfft32x4{4096});
    timeit("radix2<complex32x4, N>()", 4096, 4, cfft32x4_fixed<4096>{});
#endif
    std::printf("\n");

    timeit("radix2<complex<float>>(N)", 8192, 1, cfft<float>{8192});
    timeit("radix2<complex<float>, N>()", 8192, 1, cfft_fixed<float, 8192>{});
#ifdef __AVX__
    timeit("radix2<complex32x4>(N)", 8192, 4, cfft32x4{8192});
    timeit("radix2<complex32x4, N>()", 8192, 4, cfft32x4_fixed<8192>{});
#endif

    return EXIT_SUCCESS;
}
