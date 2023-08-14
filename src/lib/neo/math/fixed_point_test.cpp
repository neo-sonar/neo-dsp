#include "fixed_point.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <span>
#include <utility>
#include <vector>

[[nodiscard]] static auto approx_equal(float a, float b, float tolerance) -> bool
{
    return std::abs(a - b) < tolerance;
}

template<typename T>
static constexpr auto tolerance = [] {
    if constexpr (std::same_as<T, neo::q7>) {
        return 0.01F;
    }
    return 0.0001F;
}();

TEMPLATE_TEST_CASE("neo/math/fixed_point: to_float(fixed_point)", "", neo::q7, neo::q15)
{
    using fixed_point_t = TestType;

    REQUIRE(approx_equal(to_float(fixed_point_t{0.00F}), 0.00F, tolerance<fixed_point_t>));
    REQUIRE(approx_equal(to_float(fixed_point_t{0.12F}), 0.12F, tolerance<fixed_point_t>));
    REQUIRE(approx_equal(to_float(fixed_point_t{0.25F}), 0.25F, tolerance<fixed_point_t>));
    REQUIRE(approx_equal(to_float(fixed_point_t{0.33F}), 0.33F, tolerance<fixed_point_t>));
    REQUIRE(approx_equal(to_float(fixed_point_t{0.40F}), 0.40F, tolerance<fixed_point_t>));
    REQUIRE(approx_equal(to_float(fixed_point_t{0.50F}), 0.50F, tolerance<fixed_point_t>));
    REQUIRE(approx_equal(to_float(fixed_point_t{0.75F}), 0.75F, tolerance<fixed_point_t>));
}

TEMPLATE_TEST_CASE("neo/math/fixed_point: unary_op(fixed_point)", "", neo::q7, neo::q15)
{
    using fixed_point_t = TestType;

    auto unary_op = [](auto val, auto op) {
        auto const result   = to_float(op(fixed_point_t{val}));
        auto const expected = op(val);
        return approx_equal(result, expected, tolerance<fixed_point_t>);
    };

    // operator+
    auto const unary_plus = [](auto val) { return +val; };
    REQUIRE(unary_op(0.00F, unary_plus));
    REQUIRE(unary_op(0.12F, unary_plus));
    REQUIRE(unary_op(0.25F, unary_plus));
    REQUIRE(unary_op(0.33F, unary_plus));
    REQUIRE(unary_op(0.40F, unary_plus));
    REQUIRE(unary_op(0.45F, unary_plus));
    REQUIRE(unary_op(0.49F, unary_plus));
    REQUIRE(unary_op(0.75F, unary_plus));
    REQUIRE(unary_op(0.99F, unary_plus));

    REQUIRE(unary_op(-0.00F, unary_plus));
    REQUIRE(unary_op(-0.12F, unary_plus));
    REQUIRE(unary_op(-0.25F, unary_plus));
    REQUIRE(unary_op(-0.33F, unary_plus));
    REQUIRE(unary_op(-0.40F, unary_plus));
    REQUIRE(unary_op(-0.45F, unary_plus));
    REQUIRE(unary_op(-0.49F, unary_plus));
    REQUIRE(unary_op(-0.75F, unary_plus));
    REQUIRE(unary_op(-0.99F, unary_plus));

    // operator-
    REQUIRE(unary_op(0.00F, std::negate()));
    REQUIRE(unary_op(0.12F, std::negate()));
    REQUIRE(unary_op(0.25F, std::negate()));
    REQUIRE(unary_op(0.33F, std::negate()));
    REQUIRE(unary_op(0.40F, std::negate()));
    REQUIRE(unary_op(0.45F, std::negate()));
    REQUIRE(unary_op(0.49F, std::negate()));
    REQUIRE(unary_op(0.75F, std::negate()));
    REQUIRE(unary_op(0.99F, std::negate()));

    REQUIRE(unary_op(-0.00F, std::negate()));
    REQUIRE(unary_op(-0.12F, std::negate()));
    REQUIRE(unary_op(-0.25F, std::negate()));
    REQUIRE(unary_op(-0.33F, std::negate()));
    REQUIRE(unary_op(-0.40F, std::negate()));
    REQUIRE(unary_op(-0.45F, std::negate()));
    REQUIRE(unary_op(-0.49F, std::negate()));
    REQUIRE(unary_op(-0.75F, std::negate()));
    REQUIRE(unary_op(-0.99F, std::negate()));
}

TEMPLATE_TEST_CASE("neo/math/fixed_point: binary_op(fixed_point)", "", neo::q7, neo::q15)
{
    using fixed_point_t = TestType;

    auto binary_op = [](auto lhs, auto rhs, auto op) {
        auto const result   = to_float(op(fixed_point_t{lhs}, fixed_point_t{rhs}));
        auto const expected = op(lhs, rhs);
        return approx_equal(result, expected, tolerance<fixed_point_t>);
    };

    // operator+
    REQUIRE(binary_op(0.00F, 0.5F, std::plus()));
    REQUIRE(binary_op(0.12F, 0.5F, std::plus()));
    REQUIRE(binary_op(0.25F, 0.5F, std::plus()));
    REQUIRE(binary_op(0.33F, 0.5F, std::plus()));
    REQUIRE(binary_op(0.40F, 0.5F, std::plus()));
    REQUIRE(binary_op(0.45F, 0.5F, std::plus()));
    REQUIRE(binary_op(0.49F, 0.5F, std::plus()));

    // operator-
    REQUIRE(binary_op(0.00F, 0.5F, std::minus()));
    REQUIRE(binary_op(0.12F, 0.5F, std::minus()));
    REQUIRE(binary_op(0.25F, 0.5F, std::minus()));
    REQUIRE(binary_op(0.33F, 0.5F, std::minus()));
    REQUIRE(binary_op(0.40F, 0.5F, std::minus()));
    REQUIRE(binary_op(0.45F, 0.5F, std::minus()));
    REQUIRE(binary_op(0.49F, 0.5F, std::minus()));

    // operator*
    REQUIRE(binary_op(0.00F, 0.5F, std::multiplies()));
    REQUIRE(binary_op(0.12F, 0.5F, std::multiplies()));
    REQUIRE(binary_op(0.25F, 0.5F, std::multiplies()));
    REQUIRE(binary_op(0.33F, 0.5F, std::multiplies()));
    REQUIRE(binary_op(0.40F, 0.5F, std::multiplies()));
    REQUIRE(binary_op(0.45F, 0.5F, std::multiplies()));
    REQUIRE(binary_op(0.49F, 0.5F, std::multiplies()));
}

TEMPLATE_TEST_CASE("neo/math/fixed_point: comparison(fixed_point)", "", neo::q7, neo::q15)
{
    using fixed_point_t = TestType;

    auto compare_op = [](auto lhs, auto rhs, auto op) {
        auto const result   = op(fixed_point_t{lhs}, fixed_point_t{rhs});
        auto const expected = op(lhs, rhs);
        return result == expected;
    };

    // operator==
    REQUIRE(compare_op(+0.00F, +0.00F, std::equal_to()));
    REQUIRE(compare_op(+0.50F, +0.00F, std::equal_to()));
    REQUIRE(compare_op(+0.50F, -0.50F, std::equal_to()));
    REQUIRE(compare_op(+0.50F, +0.50F, std::equal_to()));

    // operator!=
    REQUIRE(compare_op(+0.00F, +0.00F, std::not_equal_to()));
    REQUIRE(compare_op(+0.50F, +0.00F, std::not_equal_to()));
    REQUIRE(compare_op(+0.50F, -0.50F, std::not_equal_to()));
    REQUIRE(compare_op(+0.50F, +0.50F, std::not_equal_to()));

    // operator<
    REQUIRE(compare_op(+0.00F, +0.00F, std::less()));
    REQUIRE(compare_op(+0.50F, +0.00F, std::less()));
    REQUIRE(compare_op(+0.50F, -0.50F, std::less()));
    REQUIRE(compare_op(+0.50F, +0.50F, std::less()));

    // operator<=
    REQUIRE(compare_op(+0.00F, +0.00F, std::less_equal()));
    REQUIRE(compare_op(+0.50F, +0.00F, std::less_equal()));
    REQUIRE(compare_op(+0.50F, -0.50F, std::less_equal()));
    REQUIRE(compare_op(+0.50F, +0.50F, std::less_equal()));

    // operator>
    REQUIRE(compare_op(+0.00F, +0.00F, std::greater()));
    REQUIRE(compare_op(+0.50F, +0.00F, std::greater()));
    REQUIRE(compare_op(+0.50F, -0.50F, std::greater()));
    REQUIRE(compare_op(+0.50F, +0.50F, std::greater()));

    // operator>=
    REQUIRE(compare_op(+0.00F, +0.00F, std::greater_equal()));
    REQUIRE(compare_op(+0.50F, +0.00F, std::greater_equal()));
    REQUIRE(compare_op(+0.50F, -0.50F, std::greater_equal()));
    REQUIRE(compare_op(+0.50F, +0.50F, std::greater_equal()));
}

TEMPLATE_TEST_CASE("neo/math/fixed_point: add(fixed_point, fixed_point)", "", neo::q7, neo::q15)
{
    using fxp_t = TestType;

    {
        // empty
        auto lhs = std::vector<fxp_t>();
        auto rhs = std::vector<fxp_t>();
        auto out = std::vector<fxp_t>();
        neo::add(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});
    }

    {
        // scalar
        auto lhs = std::vector<fxp_t>{fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}};
        auto rhs = std::vector<fxp_t>{fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}};
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::add(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.5F + 0.25F, tolerance<TestType>); };
        REQUIRE(std::all_of(out.begin(), out.end(), eq));
    }

    {
        // vectorized
        auto lhs = std::vector<fxp_t>(1029, fxp_t{0.125F});
        auto rhs = std::vector<fxp_t>(1029, fxp_t{0.25F});
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::add(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.125F + 0.25F, tolerance<TestType>); };
        REQUIRE(std::all_of(out.begin(), out.end(), eq));
    }
}

TEMPLATE_TEST_CASE("neo/math/fixed_point: subtract(fixed_point, fixed_point)", "", neo::q7, neo::q15)
{
    using fxp_t = TestType;

    {
        // empty
        auto lhs = std::vector<fxp_t>();
        auto rhs = std::vector<fxp_t>();
        auto out = std::vector<fxp_t>();
        neo::subtract(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});
    }

    {
        // scalar
        auto lhs = std::vector<fxp_t>{fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}};
        auto rhs = std::vector<fxp_t>{fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}};
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::subtract(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.5F - 0.25F, tolerance<TestType>); };
        REQUIRE(std::all_of(out.begin(), out.end(), eq));
    }

    {
        // vectorized
        auto lhs = std::vector<fxp_t>(1029, fxp_t{0.125F});
        auto rhs = std::vector<fxp_t>(1029, fxp_t{0.25F});
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::subtract(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.125F - 0.25F, tolerance<TestType>); };
        REQUIRE(std::all_of(out.begin(), out.end(), eq));
    }
}

TEMPLATE_TEST_CASE("neo/math/fixed_point: multiply(fixed_point, fixed_point)", "", neo::q7, neo::q15)
{
    using fxp_t = TestType;

    {
        // empty
        auto lhs = std::vector<fxp_t>();
        auto rhs = std::vector<fxp_t>();
        auto out = std::vector<fxp_t>();
        neo::multiply(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});
    }

    {
        // scalar
        auto lhs = std::vector<fxp_t>{fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}};
        auto rhs = std::vector<fxp_t>{fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}};
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::multiply(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.5F * 0.25F, tolerance<TestType>); };
        REQUIRE(std::all_of(out.begin(), out.end(), eq));
    }

    {
        // vectorized
        auto lhs = std::vector<fxp_t>(1029, fxp_t{0.125F});
        auto rhs = std::vector<fxp_t>(1029, fxp_t{0.25F});
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::multiply(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.125F * 0.25F, tolerance<TestType>); };
        REQUIRE(std::all_of(out.begin(), out.end(), eq));
    }
}

TEST_CASE("neo/math/fixed_point: complex_q15")
{
    STATIC_REQUIRE(neo::complex<neo::complex_q15>);
    STATIC_REQUIRE(neo::is_complex<neo::complex_q15>);

    auto const lhs = neo::complex_q15{
        neo::q15{neo::underlying_value, 1430},
        neo::q15{neo::underlying_value,  420}
    };
    auto const rhs = neo::complex_q15{
        neo::q15{neo::underlying_value, 1000},
        neo::q15{neo::underlying_value,  100}
    };

    auto const sum = lhs + rhs;
    REQUIRE(sum.real().value() == 2430);
    REQUIRE(sum.imag().value() == 520);

    auto const diff = lhs - rhs;
    REQUIRE(diff.real().value() == 430);
    REQUIRE(diff.imag().value() == 320);

    auto const product = lhs * rhs;
    REQUIRE(product.real().value() == 9);
    REQUIRE(product.imag().value() == 7);
}

template<typename Batch>
static auto test_simd_fixed_point()
{
    using Scalar = Batch::value_type;
    using Int    = Scalar::storage_type;

    auto lhs = std::array<Scalar, Batch::size>{};
    auto rhs = std::array<Scalar, Batch::size>{};
    auto out = std::array<Scalar, Batch::size>{};

    std::fill(lhs.begin(), lhs.end(), Scalar{neo::underlying_value, Int(50)});
    std::fill(rhs.begin(), rhs.end(), Scalar{neo::underlying_value, Int(42)});
    auto const l = Batch::load_unaligned(lhs.data());
    auto const r = Batch::load_unaligned(rhs.data());

    auto const sum = l + r;
    sum.store_unaligned(out.data());
    for (auto const val : out) {
        REQUIRE(val.value() == Int(92));
    }

    auto const diff = l - r;
    diff.store_unaligned(out.data());
    for (auto const val : out) {
        REQUIRE(val.value() == Int(8));
    }

    auto const brodcasted = Batch::broadcast(Scalar{neo::underlying_value, Int(123)});
    brodcasted.store_unaligned(out.data());
    for (auto const val : out) {
        REQUIRE(val.value() == Int(123));
    }
}

#if defined(NEO_HAS_SIMD_SSE2)

TEMPLATE_TEST_CASE("neo/math: fixed_point_batch", "", neo::simd::q7x16, neo::simd::q15x8)
{
    STATIC_REQUIRE(TestType::alignment == 16);
    STATIC_REQUIRE(TestType::size == 16 / sizeof(typename TestType::value_type));
    test_simd_fixed_point<TestType>();
}

#endif

#if defined(NEO_HAS_SIMD_AVX2)

TEMPLATE_TEST_CASE("neo/math: fixed_point_batch", "", neo::simd::q7x32, neo::simd::q15x16)
{
    STATIC_REQUIRE(TestType::alignment == 32);
    STATIC_REQUIRE(TestType::size == 32 / sizeof(typename TestType::value_type));
    test_simd_fixed_point<TestType>();
}

#endif
