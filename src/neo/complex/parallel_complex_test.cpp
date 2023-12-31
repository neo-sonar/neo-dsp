// SPDX-License-Identifier: MIT

#include "parallel_complex.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <array>

template<typename ComplexBatch>
static auto test()
{
    using Complex     = ComplexBatch;
    using Float       = typename Complex::batch_type;
    using ScalarFloat = typename Complex::real_scalar_type;

    STATIC_REQUIRE(neo::complex<ComplexBatch>);
    STATIC_REQUIRE(neo::is_complex<ComplexBatch>);

    SECTION("add")
    {
        auto const lhs = Complex{Float::broadcast(ScalarFloat(1)), Float::broadcast(ScalarFloat(2))};
        auto const rhs = Complex{Float::broadcast(ScalarFloat(2)), Float::broadcast(ScalarFloat(4))};
        auto const sum = lhs + rhs;

        auto real  = sum.real();
        auto reals = std::array<ScalarFloat, Complex::size>{};
        real.store_unaligned(reals.data());

        auto imag  = sum.imag();
        auto imags = std::array<ScalarFloat, Complex::size>{};
        imag.store_unaligned(imags.data());

        for (auto r : reals) {
            REQUIRE(r == Catch::Approx(ScalarFloat(3)));
        }

        for (auto i : imags) {
            REQUIRE(i == Catch::Approx(ScalarFloat(6)));
        }
    }

    SECTION("sub")
    {
        auto const lhs  = Complex{Float::broadcast(ScalarFloat(1)), Float::broadcast(ScalarFloat(2))};
        auto const rhs  = Complex{Float::broadcast(ScalarFloat(2)), Float::broadcast(ScalarFloat(4))};
        auto const diff = lhs - rhs;

        auto real  = diff.real();
        auto reals = std::array<ScalarFloat, Complex::size>{};
        real.store_unaligned(reals.data());

        auto imag  = diff.imag();
        auto imags = std::array<ScalarFloat, Complex::size>{};
        imag.store_unaligned(imags.data());

        for (auto r : reals) {
            REQUIRE(r == Catch::Approx(ScalarFloat(-1)));
        }

        for (auto i : imags) {
            REQUIRE(i == Catch::Approx(ScalarFloat(-2)));
        }
    }

    SECTION("mul")
    {
        auto const lhs  = Complex{Float::broadcast(ScalarFloat(1)), Float::broadcast(ScalarFloat(2))};
        auto const rhs  = Complex{Float::broadcast(ScalarFloat(3)), Float::broadcast(ScalarFloat(4))};
        auto const diff = lhs * rhs;

        auto real  = diff.real();
        auto reals = std::array<ScalarFloat, Complex::size>{};
        real.store_unaligned(reals.data());

        auto imag  = diff.imag();
        auto imags = std::array<ScalarFloat, Complex::size>{};
        imag.store_unaligned(imags.data());

        for (auto r : reals) {
            REQUIRE(r == Catch::Approx(ScalarFloat(-5)));
        }

        for (auto i : imags) {
            REQUIRE(i == Catch::Approx(ScalarFloat(10)));
        }
    }
}

#if defined(NEO_HAS_BUILTIN_FLOAT16) and defined(NEO_HAS_SIMD_F16C)
TEMPLATE_TEST_CASE("neo/complex: parallel_complex", "", neo::pcomplex32x8, neo::pcomplex32x16) { test<TestType>(); }
#endif

#if defined(NEO_HAS_SIMD_SSE2)
TEMPLATE_TEST_CASE("neo/complex: parallel_complex", "", neo::pcomplex64x4, neo::pcomplex128x2) { test<TestType>(); }
#endif

#if defined(NEO_HAS_SIMD_AVX)
TEMPLATE_TEST_CASE("neo/complex: parallel_complex", "", neo::pcomplex64x8, neo::pcomplex128x4) { test<TestType>(); }
#endif

#if defined(NEO_HAS_SIMD_AVX512F)
TEMPLATE_TEST_CASE("neo/complex: parallel_complex", "", neo::pcomplex64x16, neo::pcomplex128x8) { test<TestType>(); }
#endif
