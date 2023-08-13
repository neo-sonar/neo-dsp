#include "simd.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>

#if defined(NEO_HAS_SIMD_SSE2)

TEST_CASE("neo/math: simd::float32x4")
{
    auto output = std::array<float, 4>{};
    auto reg    = neo::simd::float32x4::broadcast(1.0F);
    reg.store_unaligned(output.data());
    for (auto val : output) {
        REQUIRE(val == Catch::Approx(1.0));
    }
}

TEST_CASE("neo/math: simd::float64x2")
{
    auto output = std::array<double, 2>{};
    auto reg    = neo::simd::float64x2::broadcast(1.0);
    reg.store_unaligned(output.data());
    for (auto val : output) {
        REQUIRE(val == Catch::Approx(1.0));
    }
}

using complex_types = std::tuple<
    #if defined(NEO_HAS_SIMD_AVX512F)
    neo::simd::complex32x8,
    neo::simd::complex64x4,
    #endif
    #if defined(NEO_HAS_SIMD_AVX)
    neo::simd::complex32x4,
    neo::simd::complex64x2,
    #endif
    neo::simd::complex32x2,
    neo::simd::complex64x1>;

TEMPLATE_LIST_TEST_CASE("neo/math: arithmetic", "", complex_types)
{
    using Complex    = TestType;
    using FloatBatch = typename Complex::batch_type;
    using Float      = typename FloatBatch::value_type;

    auto test = [](auto op, auto left_val, auto right_val, auto expected) -> void {
        static constexpr auto const size = Complex::size;

        auto left = std::array<std::complex<Float>, size>{};
        std::fill(left.begin(), left.end(), left_val);

        auto right = std::array<std::complex<Float>, size>{};
        std::fill(right.begin(), right.end(), right_val);

        auto lhs    = Complex::load_unaligned(left.data());
        auto rhs    = Complex::load_unaligned(right.data());
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

#endif
