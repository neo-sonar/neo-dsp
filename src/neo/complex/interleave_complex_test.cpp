// SPDX-License-Identifier: MIT
#include "interleave_complex.hpp"

#include <neo/math/float_equality.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <array>

#if defined(NEO_HAS_SIMD_SSE2)

TEST_CASE("neo/complex: simd::cmul(sse2/sse3)")
{
    using Complex  = neo::icomplex64x2;
    using Register = neo::icomplex64x2::register_type;

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
    test([](auto lhs, auto rhs) {
        return Complex{neo::cmul(static_cast<Register>(lhs.batch()), static_cast<Register>(rhs.batch()))};
    });
    // TODO Add cmul for sse2
    test([](auto lhs, auto rhs) {
        return Complex{neo::detail::cmul_sse2(static_cast<Register>(lhs.batch()), static_cast<Register>(rhs.batch()))};
    });
}

#endif

namespace {
template<typename T>
struct testing_complex
{
    using value_type = T;

    testing_complex() = default;

    testing_complex(T re) : _data{re} {}

    testing_complex(T re, T im) : _data{re, im} {}

    [[nodiscard]] auto real() const noexcept { return _data[0]; }

    [[nodiscard]] auto imag() const noexcept { return _data[1]; }

private:
    std::array<T, 2> _data;
};
}  // namespace

template<typename T>
inline constexpr auto const neo::is_complex<testing_complex<T>> = true;

template<typename ComplexBatch>
static auto test()
{
    using FloatBatch = typename ComplexBatch::batch_type;
    using Float      = typename FloatBatch::value_type;

    STATIC_REQUIRE(neo::complex<ComplexBatch>);
    STATIC_REQUIRE(neo::is_complex<ComplexBatch>);

    auto test = [](auto op, auto left_val, auto right_val, auto expected) -> void {
        static constexpr auto const size = ComplexBatch::size;

        auto left = std::array<testing_complex<Float>, size>{};
        std::fill(left.begin(), left.end(), left_val);

        auto right = std::array<testing_complex<Float>, size>{};
        std::fill(right.begin(), right.end(), right_val);

        auto lhs    = ComplexBatch::load_unaligned(left.data());
        auto rhs    = ComplexBatch::load_unaligned(right.data());
        auto result = op(lhs, rhs);

        auto output = std::array<testing_complex<Float>, size>{};
        result.store_unaligned(output.data());
        for (auto val : output) {
            REQUIRE(val.real() == Catch::Approx(expected.real()));
            REQUIRE(val.imag() == Catch::Approx(expected.imag()));
        }
    };

    auto const ones   = testing_complex{Float(1), Float(1)};
    auto const nones  = testing_complex{Float(-1), Float(-1)};
    auto const twos   = testing_complex{Float(2), Float(2)};
    auto const threes = testing_complex{Float(3), Float(3)};

    test(std::plus{}, ones, twos, threes);
    test(std::minus{}, ones, twos, nones);
}

#if defined(NEO_HAS_SIMD_SSE2)
TEMPLATE_TEST_CASE("neo/complex: interleave_complex", "", neo::icomplex64x2, neo::icomplex128x1) { test<TestType>(); }
#endif

#if defined(NEO_HAS_SIMD_AVX)
TEMPLATE_TEST_CASE("neo/complex: interleave_complex", "", neo::icomplex64x4, neo::icomplex128x2) { test<TestType>(); }
#endif

#if defined(NEO_HAS_SIMD_AVX512F)
TEMPLATE_TEST_CASE("neo/complex: interleave_complex", "", neo::icomplex64x8, neo::icomplex128x4) { test<TestType>(); }
#endif
