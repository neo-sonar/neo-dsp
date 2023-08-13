#include "simd.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>

#if defined(NEO_HAS_SIMD_SSE2)

TEST_CASE("neo/math: simd::cmul(sse3)")
{
    using Complex = neo::simd::complex32x2;

    auto test = [](auto op) {
        auto const lhs_vals = std::array{
            std::complex{1.0F, 2.0F},
            std::complex{5.0F, 6.0F}
        };

        auto const rhs_vals = std::array{
            std::complex{3.0F, 4.0F},
            std::complex{7.0F, 8.0F}
        };

        auto const lhs     = Complex::load_unaligned(lhs_vals.data());
        auto const rhs     = Complex::load_unaligned(rhs_vals.data());
        auto const product = op(lhs, rhs);

        auto output = std::array<std::complex<float>, Complex::size>{};
        product.store_unaligned(output.data());

        REQUIRE(output[0].real() == Catch::Approx(-5.0));
        REQUIRE(output[0].imag() == Catch::Approx(10.0));

        REQUIRE(output[1].real() == Catch::Approx(-13.0));
        REQUIRE(output[1].imag() == Catch::Approx(82.0));
    };

    test([](auto lhs, auto rhs) { return lhs * rhs; });
    test([](auto lhs, auto rhs) { return Complex{neo::simd::cmul(lhs, rhs)}; });
    // TODO Add cmul for sse2
    // test([](auto lhs, auto rhs) { return Complex{neo::simd::detail::cmul_sse2(lhs, rhs)}; });
}

#endif

template<typename Batch>
static auto test_float_batch()
{
    using Float = typename Batch::value_type;

    auto const reg = Batch::broadcast(Float(42));
    auto output    = std::array<Float, Batch::size>{};
    reg.store_unaligned(output.data());

    for (auto val : output) {
        REQUIRE(val == Catch::Approx(42.0));
    }
}

template<typename ComplexBatch>
static auto test_complex_batch()
{
    using FloatBatch = typename ComplexBatch::batch_type;
    using Float      = typename FloatBatch::value_type;

    auto test = [](auto op, auto left_val, auto right_val, auto expected) -> void {
        static constexpr auto const size = ComplexBatch::size;

        auto left = std::array<std::complex<Float>, size>{};
        std::fill(left.begin(), left.end(), left_val);

        auto right = std::array<std::complex<Float>, size>{};
        std::fill(right.begin(), right.end(), right_val);

        auto lhs    = ComplexBatch::load_unaligned(left.data());
        auto rhs    = ComplexBatch::load_unaligned(right.data());
        auto result = op(lhs, rhs);

        auto output = std::array<std::complex<Float>, size>{};
        result.store_unaligned(output.data());
        for (auto val : output) {
            REQUIRE(val.real() == Catch::Approx(expected.real()));
            REQUIRE(val.imag() == Catch::Approx(expected.imag()));
        }
    };

    auto const ones   = std::complex{Float(1), Float(1)};
    auto const twos   = std::complex{Float(2), Float(2)};
    auto const threes = std::complex{Float(3), Float(3)};

    test(std::plus{}, ones, twos, threes);
    test(std::minus{}, ones, twos, -ones);
}

#if defined(NEO_HAS_SIMD_SSE2)

TEMPLATE_TEST_CASE("neo/math: float_batch", "", neo::simd::float32x4, neo::simd::float64x2)
{
    test_float_batch<TestType>();
}

TEMPLATE_TEST_CASE("neo/math: complex_batch", "", neo::simd::complex32x2, neo::simd::complex64x1)
{
    test_complex_batch<TestType>();
}

#endif

#if defined(NEO_HAS_SIMD_AVX)

TEMPLATE_TEST_CASE("neo/math: float_batch", "", neo::simd::float32x8, neo::simd::float64x4)
{
    test_float_batch<TestType>();
}

TEMPLATE_TEST_CASE("neo/math: complex_batch", "", neo::simd::complex32x4, neo::simd::complex64x2)
{
    test_complex_batch<TestType>();
}

#endif

#if defined(NEO_HAS_SIMD_AVX512F)

TEMPLATE_TEST_CASE("neo/math: float_batch", "", neo::simd::float32x16, neo::simd::float64x8)
{
    test_float_batch<TestType>();
}

TEMPLATE_TEST_CASE("neo/math: complex_batch", "", neo::simd::complex32x8, neo::simd::complex64x4)
{
    test_complex_batch<TestType>();
}

#endif
