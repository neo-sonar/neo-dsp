#include "simd.hpp"

#include <neo/math/float_equality.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>

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

#if defined(NEO_HAS_SIMD_SSE2)

TEST_CASE("neo/simd: simd::cmul(sse3)")
{
    using Complex  = neo::simd::icomplex64x2;
    using Register = neo::simd::icomplex64x2::register_type;

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
        return Complex{neo::simd::cmul(static_cast<Register>(lhs.batch()), static_cast<Register>(rhs.batch()))};
    });
    // TODO Add cmul for sse2
    test([](auto lhs, auto rhs) {
        return Complex{
            neo::simd::detail::cmul_sse2(static_cast<Register>(lhs.batch()), static_cast<Register>(rhs.batch()))};
    });
}

#endif

template<typename Batch>
static auto test_float_batch()
{
    using Float = typename Batch::value_type;

    auto test = [](auto op, auto left_scalar, auto right_scalar, auto expected) {
        auto left  = std::array<Float, Batch::size>{};
        auto right = std::array<Float, Batch::size>{};

        std::fill(left.begin(), left.end(), left_scalar);
        std::fill(right.begin(), right.end(), right_scalar);

        auto output = std::array<Float, Batch::size>{};
        auto result = op(Batch::load_unaligned(left.data()), Batch::load_unaligned(right.data()));
        result.store_unaligned(output.data());

        for (auto val : output) {
            REQUIRE(val == Catch::Approx(expected));
        }
    };

    test(std::plus{}, Float(1), Float(2), Float(3));
    test(std::minus{}, Float(1), Float(2), Float(-1));
    test(std::multiplies{}, Float(1), Float(2), Float(2));
}

template<typename ComplexBatch>
static auto test_icomplex_batch()
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

template<typename ComplexBatch>
static auto test_pcomplex_batch()
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

#if defined(NEO_HAS_SIMD_SSE2)

TEMPLATE_TEST_CASE("neo/simd: float_batch", "", neo::simd::float32x4, neo::simd::float64x2)
{
    test_float_batch<TestType>();
}

TEMPLATE_TEST_CASE("neo/simd: complex_batch", "", neo::simd::icomplex64x2, neo::simd::icomplex128x1)
{
    test_icomplex_batch<TestType>();
}

static_assert(std::same_as<typename neo::simd::pcomplex64x4::batch_type, neo::simd::float32x4>);
static_assert(std::same_as<typename neo::simd::pcomplex64x4::register_type, __m128>);
static_assert(std::same_as<typename neo::simd::pcomplex64x4::real_scalar_type, float>);

static_assert(std::same_as<typename neo::simd::pcomplex128x2::batch_type, neo::simd::float64x2>);
static_assert(std::same_as<typename neo::simd::pcomplex128x2::register_type, __m128d>);
static_assert(std::same_as<typename neo::simd::pcomplex128x2::real_scalar_type, double>);

TEMPLATE_TEST_CASE("neo/simd: parallel_complex_batch", "", neo::simd::pcomplex64x4, neo::simd::pcomplex128x2)
{
    test_pcomplex_batch<TestType>();
}

#endif

#if defined(NEO_HAS_SIMD_AVX)

TEMPLATE_TEST_CASE("neo/simd: float_batch", "", neo::simd::float32x8, neo::simd::float64x4)
{
    test_float_batch<TestType>();
}

TEMPLATE_TEST_CASE("neo/simd: complex_batch", "", neo::simd::icomplex64x4, neo::simd::icomplex128x2)
{
    test_icomplex_batch<TestType>();
}

static_assert(std::same_as<typename neo::simd::pcomplex64x8::batch_type, neo::simd::float32x8>);
static_assert(std::same_as<typename neo::simd::pcomplex64x8::register_type, __m256>);
static_assert(std::same_as<typename neo::simd::pcomplex64x8::real_scalar_type, float>);

static_assert(std::same_as<typename neo::simd::pcomplex128x4::batch_type, neo::simd::float64x4>);
static_assert(std::same_as<typename neo::simd::pcomplex128x4::register_type, __m256d>);
static_assert(std::same_as<typename neo::simd::pcomplex128x4::real_scalar_type, double>);

TEMPLATE_TEST_CASE("neo/simd: parallel_complex_batch", "", neo::simd::pcomplex64x8, neo::simd::pcomplex128x4)
{
    test_pcomplex_batch<TestType>();
}

#endif

#if defined(NEO_HAS_BASIC_FLOAT16)
TEMPLATE_TEST_CASE("neo/simd: float_batch", "", neo::simd::float16x8, neo::simd::float16x16)
{
    test_float_batch<TestType>();
}

// TEMPLATE_TEST_CASE("neo/simd: complex_batch", "", neo::simd::icomplex32x4) { test_icomplex_batch<TestType>(); }

TEMPLATE_TEST_CASE("neo/simd: parallel_complex_batch", "", neo::simd::pcomplex32x8, neo::simd::pcomplex32x16)
{
    test_pcomplex_batch<TestType>();
}
#endif

#if defined(NEO_HAS_SIMD_AVX512F)

TEMPLATE_TEST_CASE("neo/simd: float_batch", "", neo::simd::float32x16, neo::simd::float64x8)
{
    test_float_batch<TestType>();
}

TEMPLATE_TEST_CASE("neo/simd: complex_batch", "", neo::simd::icomplex64x8, neo::simd::icomplex128x4)
{
    test_icomplex_batch<TestType>();
}

static_assert(std::same_as<typename neo::simd::pcomplex64x16::batch_type, neo::simd::float32x16>);
static_assert(std::same_as<typename neo::simd::pcomplex64x16::register_type, __m512>);
static_assert(std::same_as<typename neo::simd::pcomplex64x16::real_scalar_type, float>);

static_assert(std::same_as<typename neo::simd::pcomplex128x8::batch_type, neo::simd::float64x8>);
static_assert(std::same_as<typename neo::simd::pcomplex128x8::register_type, __m512d>);
static_assert(std::same_as<typename neo::simd::pcomplex128x8::real_scalar_type, double>);

TEMPLATE_TEST_CASE("neo/simd: parallel_complex_batch", "", neo::simd::pcomplex64x16, neo::simd::pcomplex128x8)
{
    test_pcomplex_batch<TestType>();
}

#endif
