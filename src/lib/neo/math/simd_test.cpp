#include "simd.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>

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

    // auto const identity = std::complex{Float(1), Float(0)};
    // test(std::multiplies{}, threes, identity, threes);
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
