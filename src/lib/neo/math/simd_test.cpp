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
    auto test = [](auto op, auto expected) -> void {
        using Complex    = TestType;
        using FloatBatch = typename Complex::batch_type;
        using Float      = typename FloatBatch::value_type;

        static constexpr auto const size = FloatBatch::batch_size;

        auto left = std::array<std::complex<Float>, Complex::batch_size>{};
        std::fill(left.begin(), left.end(), std::complex{Float(1), Float(1)});

        auto right = std::array<std::complex<Float>, Complex::batch_size>{};
        std::fill(right.begin(), right.end(), std::complex{Float(2), Float(2)});

        auto lhs    = Complex::load_unaligned(left.data());
        auto rhs    = Complex::load_unaligned(right.data());
        auto result = op(lhs, rhs);

        auto output = std::array<Float, size>{};
        result.store_unaligned(output.data());
        for (auto val : output) {
            REQUIRE(val == Catch::Approx(expected));
        }
    };

    test(std::plus{}, 3.0);
    test(std::minus{}, -1.0);
}

#endif
