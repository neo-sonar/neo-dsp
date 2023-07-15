#include "neo/convolution/math/fixed_point.hpp"

#include "benchmark.hpp"

#include <cfloat>
#include <random>
#include <vector>

template<typename Float, std::size_t Size>
struct floating_point_mul_bench
{
    explicit floating_point_mul_bench() : _lhs(Size), _rhs(Size), _out(Size)
    {
        auto rng  = std::mt19937{std::random_device{}()};
        auto dist = std::uniform_real_distribution<Float>{Float(-1.0), Float(1.0)};
        std::generate(_lhs.begin(), _lhs.end(), [&] { return dist(rng); });
        std::generate(_rhs.begin(), _rhs.end(), [&] { return dist(rng); });
    }

    auto operator()() -> void
    {
        std::fill(_out.begin(), _out.end(), 1.0F);
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
        static constexpr auto one = FixedPoint{1.0F};
        std::fill(_out.begin(), _out.end(), one);

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

auto main() -> int
{
    using neo::fft::q15_t;
    using neo::fft::q7_t;

    neo::fft::timeit("mul(double[32768], double[32768])  :", 32768, floating_point_mul_bench<double, 32768U>{});
    neo::fft::timeit("mul(double[262144], double[262144]):", 262144, floating_point_mul_bench<double, 262144U>{});

    neo::fft::timeit("mul(float[32768], float[32768])    :", 32768, floating_point_mul_bench<float, 32768U>{});
    neo::fft::timeit("mul(float[262144], float[262144])  :", 262144, floating_point_mul_bench<float, 262144U>{});

    neo::fft::timeit("mul(q15_t[32768], q15_t[32768])    :", 32768, fixed_point_mul_bench<q15_t, 32768U>{});
    neo::fft::timeit("mul(q15_t[262144], q15_t[262144])  :", 262144, fixed_point_mul_bench<q15_t, 262144U>{});

    neo::fft::timeit("mul(q7_t[32768], q7_t[32768])      :", 32768, fixed_point_mul_bench<q7_t, 32768U>{});
    neo::fft::timeit("mul(q7_t[262144], q7_t[262144])    :", 262144, fixed_point_mul_bench<q7_t, 262144U>{});

    return EXIT_SUCCESS;
}
