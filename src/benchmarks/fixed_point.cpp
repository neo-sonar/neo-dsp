#include <neo/config.hpp>

#include <neo/algorithm.hpp>
#include <neo/container.hpp>
#include <neo/fixed_point.hpp>

#include <neo/testing/benchmark.hpp>
#include <neo/testing/testing.hpp>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <random>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace {
template<typename Func>
auto timeit(std::string_view name, size_t sizeOfT, size_t N, Func func)
{
    using microseconds = std::chrono::duration<double, std::micro>;

    auto const size       = N;
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

    auto const runs            = std::span<double>(all_runs).subspan(margin, all_runs.size() - margin * 2);
    auto const avg             = std::reduce(runs.begin(), runs.end(), 0.0) / double(runs.size());
    auto const itemsPerSec     = static_cast<int>(std::lround(double(size) / avg));
    auto const megaBytesPerSec = std::round(double(size * sizeOfT) / avg) / 1000.0;
    std::printf("%-32s avg: %.1fus - GB/sec: %.2f - N/usec: %d\n", name.data(), avg, megaBytesPerSec, itemsPerSec);
}

template<typename FloatOrComplex, std::size_t Size>
struct float_mul
{
    explicit float_mul()
        : _lhs(neo::generate_noise_signal<FloatOrComplex>(Size, std::random_device{}()))
        , _rhs(neo::generate_noise_signal<FloatOrComplex>(Size, std::random_device{}()))
        , _out(Size)
    {}

    auto operator()() noexcept -> void
    {
        neo::multiply(_lhs.to_mdspan(), _rhs.to_mdspan(), _out.to_mdspan());
        neo::do_not_optimize(_out(0));
    }

private:
    stdex::mdarray<FloatOrComplex, stdex::dextents<size_t, 1>> _lhs;
    stdex::mdarray<FloatOrComplex, stdex::dextents<size_t, 1>> _rhs;
    stdex::mdarray<FloatOrComplex, stdex::dextents<size_t, 1>> _out;
};

template<typename FloatOrComplex, typename IntOrComplex, std::size_t Size>
struct cfloat_mul
{
    explicit cfloat_mul() : _lhs(Size), _rhs(Size), _out(Size)
    {
        auto noiseA   = neo::generate_noise_signal<FloatOrComplex>(Size, std::random_device{}());
        auto noiseB   = neo::generate_noise_signal<FloatOrComplex>(Size, std::random_device{}());
        auto compress = [](FloatOrComplex val) {
            if constexpr (neo::complex<FloatOrComplex>) {
                static_assert(neo::complex<IntOrComplex>);

                using Float                 = typename FloatOrComplex::value_type;
                using Int                   = typename IntOrComplex::value_type;
                static constexpr auto scale = static_cast<Float>(std::numeric_limits<Int>::max());

                return IntOrComplex(
                    static_cast<Int>(std::lround(val.real() * scale)),
                    static_cast<Int>(std::lround(val.imag() * scale))
                );
            } else {
                static constexpr auto scale = static_cast<FloatOrComplex>(std::numeric_limits<IntOrComplex>::max());
                return static_cast<IntOrComplex>(std::lround(val * scale));
            }
        };

        std::transform(noiseA.data(), noiseA.data() + noiseA.size(), _lhs.data(), compress);
        std::transform(noiseB.data(), noiseB.data() + noiseB.size(), _rhs.data(), compress);
    }

    auto operator()() -> void
    {
        using nested_accessor = typename stdex::mdspan<IntOrComplex, stdex::dextents<size_t, 1>>::accessor_type;
        using real_accessor   = neo::compressed_accessor<FloatOrComplex, nested_accessor>;
        using extents         = stdex::dextents<size_t, 1>;
        using real_mdspan     = stdex::mdspan<FloatOrComplex, extents, stdex::layout_right, real_accessor>;

        neo::multiply(real_mdspan(_lhs.to_mdspan()), real_mdspan(_rhs.to_mdspan()), _out.to_mdspan());
    }

private:
    stdex::mdarray<IntOrComplex, stdex::dextents<size_t, 1>> _lhs;
    stdex::mdarray<IntOrComplex, stdex::dextents<size_t, 1>> _rhs;
    stdex::mdarray<FloatOrComplex, stdex::dextents<size_t, 1>> _out;
};

template<typename FixedPoint, std::size_t Size>
struct fixed_point_mul
{
    explicit fixed_point_mul() : _lhs(Size), _rhs(Size), _out(Size)
    {
        auto rng  = std::mt19937{std::random_device{}()};
        auto dist = std::uniform_real_distribution<float>{-1.0F, 1.0F};
        auto gen  = [&] {
            if constexpr (neo::complex<FixedPoint>) {
                using FP = typename FixedPoint::value_type;
                return FixedPoint{FP{dist(rng)}, FP{dist(rng)}};
            } else {
                return FixedPoint{dist(rng)};
            }
        };
        std::generate(_lhs.begin(), _lhs.end(), gen);
        std::generate(_rhs.begin(), _rhs.end(), gen);
    }

    auto operator()() -> void
    {
        if constexpr (neo::complex<FixedPoint>) {
            auto const left   = stdex::mdspan{_lhs.data(), stdex::extents{Size}};
            auto const right  = stdex::mdspan{_rhs.data(), stdex::extents{Size}};
            auto const output = stdex::mdspan{_out.data(), stdex::extents{Size}};
            neo::multiply(left, right, output);
        } else {
            auto const left   = std::span{std::as_const(_lhs)};
            auto const right  = std::span{std::as_const(_rhs)};
            auto const output = std::span{_out};
            neo::multiply(left, right, output);
        }
    }

private:
    std::vector<FixedPoint> _lhs;
    std::vector<FixedPoint> _rhs;
    std::vector<FixedPoint> _out;
};

#if defined(NEO_HAS_BASIC_FLOAT16)
template<std::size_t Size>
struct float16_mul_bench
{
    explicit float16_mul_bench() : _lhs(Size), _rhs(Size), _out(Size)
    {
        auto rng  = std::mt19937{std::random_device{}()};
        auto dist = std::uniform_real_distribution<float>{-1.0F, 1.0F};
        std::generate(_lhs.begin(), _lhs.end(), [&] { return _cvtss_sh(dist(rng), _MM_FROUND_CUR_DIRECTION); });
        std::generate(_rhs.begin(), _rhs.end(), [&] { return _cvtss_sh(dist(rng), _MM_FROUND_CUR_DIRECTION); });
    }

    auto operator()() -> void
    {
        static constexpr auto const vectorSize = 128 / 16;
        static_assert((Size % vectorSize) == 0);

        for (auto i{0}; i < Size; i += vectorSize) {
            auto const lhsWord = _mm_loadu_si128(reinterpret_cast<__m128i const*>(_lhs.data() + i));
            auto const rhsWord = _mm_loadu_si128(reinterpret_cast<__m128i const*>(_rhs.data() + i));

            auto const lhs     = _mm256_cvtph_ps(lhsWord);
            auto const rhs     = _mm256_cvtph_ps(rhsWord);
            auto const product = _mm256_mul_ps(lhs, rhs);

            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(_out.data() + i),
                _mm256_cvtps_ph(product, _MM_FROUND_CUR_DIRECTION)
            );
        }
    }

private:
    std::vector<std::uint16_t> _lhs;
    std::vector<std::uint16_t> _rhs;
    std::vector<std::uint16_t> _out;
};
#endif

}  // namespace

auto main() -> int
{
    static constexpr auto N = 131072U;

    timeit("mul(q7):     ", 1, N, fixed_point_mul<neo::q7, N>{});
    timeit("mul(q15):    ", 2, N, fixed_point_mul<neo::q15, N>{});
#if defined(NEO_HAS_BASIC_FLOAT16)
    timeit("mul(f16f32): ", 2, N, float16_mul_bench<N>{});
#endif
    timeit("mul(float):  ", 4, N, float_mul<float, N>{});
    timeit("mul(double): ", 8, N, float_mul<double, N>{});
    timeit("mul(cf8):    ", 1, N, cfloat_mul<float, int8_t, N>{});
    timeit("mul(cf16):   ", 2, N, cfloat_mul<float, int16_t, N>{});
    std::printf("\n");

    timeit("cmul(complex<cf8>):         ", 2, N, cfloat_mul<neo::complex64, neo::scalar_complex<int8_t>, N>{});
    timeit("cmul(complex<cf16>):        ", 4, N, cfloat_mul<neo::complex64, neo::scalar_complex<int16_t>, N>{});
    timeit("cmul(complex64):            ", 8, N, float_mul<neo::complex64, N>{});
    timeit("cmul(complex128):           ", 16, N, float_mul<neo::complex128, N>{});
    timeit("cmul(std::complex<float>):  ", 8, N, float_mul<std::complex<float>, N>{});
    timeit("cmul(std::complex<double>): ", 16, N, float_mul<std::complex<double>, N>{});

    return EXIT_SUCCESS;
}
