#include <neo/fixed_point.hpp>

#include <neo/testing/benchmark.hpp>
#include <neo/testing/testing.hpp>

#include <cfloat>
#include <functional>
#include <random>
#include <utility>
#include <vector>

template<typename Float, std::size_t Size>
struct floating_point_mul_bench
{
    explicit floating_point_mul_bench()
        : _lhs(neo::fft::generate_noise_signal<Float>(Size, std::random_device{}()))
        , _rhs(neo::fft::generate_noise_signal<Float>(Size, std::random_device{}()))
        , _out(Size)
    {}

    auto operator()() -> void
    {
        std::transform(_lhs.begin(), _lhs.end(), _rhs.begin(), _out.begin(), std::multiplies{});
    }

private:
    std::vector<Float> _lhs;
    std::vector<Float> _rhs;
    std::vector<Float> _out;
};

template<typename FixedPoint, std::size_t Size>
struct fixed_point_mul_bench
{
    explicit fixed_point_mul_bench() : _lhs(Size), _rhs(Size), _out(Size)
    {
        auto rng  = std::mt19937{std::random_device{}()};
        auto dist = std::uniform_real_distribution<float>{-1.0F, 1.0F};
        std::generate(_lhs.begin(), _lhs.end(), [&] { return FixedPoint{dist(rng)}; });
        std::generate(_rhs.begin(), _rhs.end(), [&] { return FixedPoint{dist(rng)}; });
    }

    auto operator()() -> void
    {
        auto const left   = std::span{std::as_const(_lhs)};
        auto const right  = std::span{std::as_const(_rhs)};
        auto const output = std::span{_out};
        neo::fft::multiply(left, right, output);
    }

private:
    std::vector<FixedPoint> _lhs;
    std::vector<FixedPoint> _rhs;
    std::vector<FixedPoint> _out;
};

#if defined(__F16C__)
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

auto main() -> int
{
    using neo::fft::q15_t;
    using neo::fft::q7_t;

    neo::fft::timeit("mul(q7_t[32768], q7_t[32768]):       ", 1, 32768, fixed_point_mul_bench<q7_t, 32768U>{});
    neo::fft::timeit("mul(q7_t[131072], q7_t[131072]):     ", 1, 131072, fixed_point_mul_bench<q7_t, 131072U>{});
    neo::fft::timeit("mul(q7_t[262144], q7_t[262144]):     ", 1, 262144, fixed_point_mul_bench<q7_t, 262144U>{});
    std::printf("\n");

    neo::fft::timeit("mul(q15_t[32768], q15_t[32768]):     ", 2, 32768, fixed_point_mul_bench<q15_t, 32768U>{});
    neo::fft::timeit("mul(q15_t[131072], q15_t[131072]):   ", 2, 131072, fixed_point_mul_bench<q15_t, 131072U>{});
    neo::fft::timeit("mul(q15_t[262144], q15_t[262144]):   ", 2, 262144, fixed_point_mul_bench<q15_t, 262144U>{});
    std::printf("\n");

#if defined(__F16C__)
    neo::fft::timeit("mul(f16f32[32768], f16f32[32768]):   ", 2, 32768, float16_mul_bench<32768U>{});
    neo::fft::timeit("mul(f16f32[131072], f16f32[131072]): ", 2, 131072, float16_mul_bench<131072U>{});
    neo::fft::timeit("mul(f16f32[262144], f16f32[262144]): ", 2, 262144, float16_mul_bench<262144U>{});
    std::printf("\n");
#endif

    neo::fft::timeit("mul(float[32768], float[32768]):     ", 4, 32768, floating_point_mul_bench<float, 32768U>{});
    neo::fft::timeit("mul(float[131072], float[131072]):   ", 4, 131072, floating_point_mul_bench<float, 131072U>{});
    neo::fft::timeit("mul(float[262144], float[262144]):   ", 4, 262144, floating_point_mul_bench<float, 262144U>{});
    std::printf("\n");

    neo::fft::timeit("mul(double[32768], double[32768]):   ", 8, 32768, floating_point_mul_bench<double, 32768U>{});
    neo::fft::timeit("mul(double[131072], double[131072]): ", 8, 131072, floating_point_mul_bench<double, 131072U>{});
    neo::fft::timeit("mul(double[262144], double[262144]): ", 8, 262144, floating_point_mul_bench<double, 262144U>{});
    std::printf("\n");

    return EXIT_SUCCESS;
}
