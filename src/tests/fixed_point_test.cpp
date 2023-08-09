#include "neo/fft/math/fixed_point.hpp"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <functional>
#include <utility>
#include <vector>

[[nodiscard]] static auto approx_equal(float a, float b, float tolerance) -> bool
{
    return std::abs(a - b) < tolerance;
}

[[nodiscard]] static auto test_q7_t_conversion() -> bool
{
    auto const tolerance = 0.01F;
    assert(approx_equal(to_float(neo::fft::q7_t{0.00F}), 0.00F, tolerance));
    assert(approx_equal(to_float(neo::fft::q7_t{0.12F}), 0.12F, tolerance));
    assert(approx_equal(to_float(neo::fft::q7_t{0.25F}), 0.25F, tolerance));
    assert(approx_equal(to_float(neo::fft::q7_t{0.33F}), 0.33F, tolerance));
    assert(approx_equal(to_float(neo::fft::q7_t{0.40F}), 0.40F, tolerance));
    assert(approx_equal(to_float(neo::fft::q7_t{0.50F}), 0.50F, tolerance));
    assert(approx_equal(to_float(neo::fft::q7_t{0.75F}), 0.75F, tolerance));
    return true;
}

[[nodiscard]] static auto test_q15_t_conversion() -> bool
{
    using fxp_t          = neo::fft::q15_t;
    auto const tolerance = 0.0001F;

    assert(approx_equal(to_float(fxp_t{0.00F}), 0.00F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.12F}), 0.12F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.25F}), 0.25F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.33F}), 0.33F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.40F}), 0.40F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.50F}), 0.50F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.75F}), 0.75F, tolerance));

    return true;
}

[[nodiscard]] static auto test_q7_t_unary_ops() -> bool
{
    using fxp_t          = neo::fft::q7_t;
    auto const tolerance = 0.01F;

    auto unary_op = [tolerance](auto val, auto op) {
        auto const result   = to_float(op(fxp_t{val}));
        auto const expected = op(val);
        return approx_equal(result, expected, tolerance);
    };

    // operator+
    auto const unary_plus = [](auto val) { return +val; };
    assert(unary_op(0.00F, unary_plus));
    assert(unary_op(0.12F, unary_plus));
    assert(unary_op(0.25F, unary_plus));
    assert(unary_op(0.33F, unary_plus));
    assert(unary_op(0.40F, unary_plus));
    assert(unary_op(0.45F, unary_plus));
    assert(unary_op(0.49F, unary_plus));
    assert(unary_op(0.75F, unary_plus));
    assert(unary_op(0.99F, unary_plus));

    assert(unary_op(-0.00F, unary_plus));
    assert(unary_op(-0.12F, unary_plus));
    assert(unary_op(-0.25F, unary_plus));
    assert(unary_op(-0.33F, unary_plus));
    assert(unary_op(-0.40F, unary_plus));
    assert(unary_op(-0.45F, unary_plus));
    assert(unary_op(-0.49F, unary_plus));
    assert(unary_op(-0.75F, unary_plus));
    assert(unary_op(-0.99F, unary_plus));

    // operator-
    assert(unary_op(0.00F, std::negate()));
    assert(unary_op(0.12F, std::negate()));
    assert(unary_op(0.25F, std::negate()));
    assert(unary_op(0.33F, std::negate()));
    assert(unary_op(0.40F, std::negate()));
    assert(unary_op(0.45F, std::negate()));
    assert(unary_op(0.49F, std::negate()));
    assert(unary_op(0.75F, std::negate()));
    assert(unary_op(0.99F, std::negate()));

    assert(unary_op(-0.00F, std::negate()));
    assert(unary_op(-0.12F, std::negate()));
    assert(unary_op(-0.25F, std::negate()));
    assert(unary_op(-0.33F, std::negate()));
    assert(unary_op(-0.40F, std::negate()));
    assert(unary_op(-0.45F, std::negate()));
    assert(unary_op(-0.49F, std::negate()));
    assert(unary_op(-0.75F, std::negate()));
    assert(unary_op(-0.99F, std::negate()));

    return true;
}

[[nodiscard]] static auto test_q15_t_unary_ops() -> bool
{
    using fxp_t          = neo::fft::q15_t;
    auto const tolerance = 0.0001F;

    auto unary_op = [tolerance](auto val, auto op) {
        auto const result   = to_float(op(fxp_t{val}));
        auto const expected = op(val);
        return approx_equal(result, expected, tolerance);
    };

    // operator+
    auto const unary_plus = [](auto val) { return +val; };
    assert(unary_op(0.00F, unary_plus));
    assert(unary_op(0.12F, unary_plus));
    assert(unary_op(0.25F, unary_plus));
    assert(unary_op(0.33F, unary_plus));
    assert(unary_op(0.40F, unary_plus));
    assert(unary_op(0.45F, unary_plus));
    assert(unary_op(0.49F, unary_plus));
    assert(unary_op(0.75F, unary_plus));
    assert(unary_op(0.99F, unary_plus));

    assert(unary_op(-0.00F, unary_plus));
    assert(unary_op(-0.12F, unary_plus));
    assert(unary_op(-0.25F, unary_plus));
    assert(unary_op(-0.33F, unary_plus));
    assert(unary_op(-0.40F, unary_plus));
    assert(unary_op(-0.45F, unary_plus));
    assert(unary_op(-0.49F, unary_plus));
    assert(unary_op(-0.75F, unary_plus));
    assert(unary_op(-0.99F, unary_plus));

    // operator-
    assert(unary_op(0.00F, std::negate()));
    assert(unary_op(0.12F, std::negate()));
    assert(unary_op(0.25F, std::negate()));
    assert(unary_op(0.33F, std::negate()));
    assert(unary_op(0.40F, std::negate()));
    assert(unary_op(0.45F, std::negate()));
    assert(unary_op(0.49F, std::negate()));
    assert(unary_op(0.75F, std::negate()));
    assert(unary_op(0.99F, std::negate()));

    assert(unary_op(-0.00F, std::negate()));
    assert(unary_op(-0.12F, std::negate()));
    assert(unary_op(-0.25F, std::negate()));
    assert(unary_op(-0.33F, std::negate()));
    assert(unary_op(-0.40F, std::negate()));
    assert(unary_op(-0.45F, std::negate()));
    assert(unary_op(-0.49F, std::negate()));
    assert(unary_op(-0.75F, std::negate()));
    assert(unary_op(-0.99F, std::negate()));

    return true;
}

[[nodiscard]] static auto test_q7_t_binary_ops() -> bool
{
    using fxp_t          = neo::fft::q7_t;
    auto const tolerance = 0.01F;

    auto binary_op = [tolerance](auto lhs, auto rhs, auto op) {
        auto const result   = to_float(op(fxp_t{lhs}, fxp_t{rhs}));
        auto const expected = op(lhs, rhs);
        return approx_equal(result, expected, tolerance);
    };

    // operator+
    assert(binary_op(0.00F, 0.5F, std::plus()));
    assert(binary_op(0.12F, 0.5F, std::plus()));
    assert(binary_op(0.25F, 0.5F, std::plus()));
    assert(binary_op(0.33F, 0.5F, std::plus()));
    assert(binary_op(0.40F, 0.5F, std::plus()));
    assert(binary_op(0.45F, 0.5F, std::plus()));
    assert(binary_op(0.49F, 0.5F, std::plus()));

    // operator-
    assert(binary_op(0.00F, 0.5F, std::minus()));
    assert(binary_op(0.12F, 0.5F, std::minus()));
    assert(binary_op(0.25F, 0.5F, std::minus()));
    assert(binary_op(0.33F, 0.5F, std::minus()));
    assert(binary_op(0.40F, 0.5F, std::minus()));
    assert(binary_op(0.45F, 0.5F, std::minus()));
    assert(binary_op(0.49F, 0.5F, std::minus()));

    // operator*
    assert(binary_op(0.00F, 0.5F, std::multiplies()));
    assert(binary_op(0.12F, 0.5F, std::multiplies()));
    assert(binary_op(0.25F, 0.5F, std::multiplies()));
    assert(binary_op(0.33F, 0.5F, std::multiplies()));
    assert(binary_op(0.40F, 0.5F, std::multiplies()));
    assert(binary_op(0.45F, 0.5F, std::multiplies()));
    assert(binary_op(0.49F, 0.5F, std::multiplies()));

    return true;
}

[[nodiscard]] static auto test_q15_t_binary_ops() -> bool
{
    using fxp_t          = neo::fft::q15_t;
    auto const tolerance = 0.0001F;

    auto binary_op = [tolerance](auto lhs, auto rhs, auto op) {
        auto const result   = to_float(op(fxp_t{lhs}, fxp_t{rhs}));
        auto const expected = op(lhs, rhs);
        return approx_equal(result, expected, tolerance);
    };

    // operator+
    assert(binary_op(0.00F, 0.5F, std::plus()));
    assert(binary_op(0.12F, 0.5F, std::plus()));
    assert(binary_op(0.25F, 0.5F, std::plus()));
    assert(binary_op(0.33F, 0.5F, std::plus()));
    assert(binary_op(0.40F, 0.5F, std::plus()));
    assert(binary_op(0.45F, 0.5F, std::plus()));
    assert(binary_op(0.49F, 0.5F, std::plus()));

    // operator-
    assert(binary_op(0.00F, 0.5F, std::minus()));
    assert(binary_op(0.12F, 0.5F, std::minus()));
    assert(binary_op(0.25F, 0.5F, std::minus()));
    assert(binary_op(0.33F, 0.5F, std::minus()));
    assert(binary_op(0.40F, 0.5F, std::minus()));
    assert(binary_op(0.45F, 0.5F, std::minus()));
    assert(binary_op(0.49F, 0.5F, std::minus()));

    // operator*
    assert(binary_op(0.00F, 0.5F, std::multiplies()));
    assert(binary_op(0.12F, 0.5F, std::multiplies()));
    assert(binary_op(0.25F, 0.5F, std::multiplies()));
    assert(binary_op(0.33F, 0.5F, std::multiplies()));
    assert(binary_op(0.40F, 0.5F, std::multiplies()));
    assert(binary_op(0.45F, 0.5F, std::multiplies()));
    assert(binary_op(0.49F, 0.5F, std::multiplies()));

    return true;
}

[[nodiscard]] static auto test_q7_t_comparison() -> bool
{
    using fxp_t = neo::fft::q7_t;

    auto compare_op = [](auto lhs, auto rhs, auto op) {
        auto const result   = op(fxp_t{lhs}, fxp_t{rhs});
        auto const expected = op(lhs, rhs);
        return result == expected;
    };

    // operator==
    assert(compare_op(+0.00F, +0.00F, std::equal_to()));
    assert(compare_op(+0.50F, +0.00F, std::equal_to()));
    assert(compare_op(+0.50F, -0.50F, std::equal_to()));
    assert(compare_op(+0.50F, +0.50F, std::equal_to()));

    // operator!=
    assert(compare_op(+0.00F, +0.00F, std::not_equal_to()));
    assert(compare_op(+0.50F, +0.00F, std::not_equal_to()));
    assert(compare_op(+0.50F, -0.50F, std::not_equal_to()));
    assert(compare_op(+0.50F, +0.50F, std::not_equal_to()));

    // operator<
    assert(compare_op(+0.00F, +0.00F, std::less()));
    assert(compare_op(+0.50F, +0.00F, std::less()));
    assert(compare_op(+0.50F, -0.50F, std::less()));
    assert(compare_op(+0.50F, +0.50F, std::less()));

    // operator<=
    assert(compare_op(+0.00F, +0.00F, std::less_equal()));
    assert(compare_op(+0.50F, +0.00F, std::less_equal()));
    assert(compare_op(+0.50F, -0.50F, std::less_equal()));
    assert(compare_op(+0.50F, +0.50F, std::less_equal()));

    // operator>
    assert(compare_op(+0.00F, +0.00F, std::greater()));
    assert(compare_op(+0.50F, +0.00F, std::greater()));
    assert(compare_op(+0.50F, -0.50F, std::greater()));
    assert(compare_op(+0.50F, +0.50F, std::greater()));

    // operator>=
    assert(compare_op(+0.00F, +0.00F, std::greater_equal()));
    assert(compare_op(+0.50F, +0.00F, std::greater_equal()));
    assert(compare_op(+0.50F, -0.50F, std::greater_equal()));
    assert(compare_op(+0.50F, +0.50F, std::greater_equal()));

    return true;
}

[[nodiscard]] static auto test_q15_t_comparison() -> bool
{
    using fxp_t = neo::fft::q15_t;

    auto compare_op = [](auto lhs, auto rhs, auto op) {
        auto const result   = op(fxp_t{lhs}, fxp_t{rhs});
        auto const expected = op(lhs, rhs);
        return result == expected;
    };

    // operator==
    assert(compare_op(+0.00F, +0.00F, std::equal_to()));
    assert(compare_op(+0.50F, +0.00F, std::equal_to()));
    assert(compare_op(+0.50F, -0.50F, std::equal_to()));
    assert(compare_op(+0.50F, +0.50F, std::equal_to()));

    // operator!=
    assert(compare_op(+0.00F, +0.00F, std::not_equal_to()));
    assert(compare_op(+0.50F, +0.00F, std::not_equal_to()));
    assert(compare_op(+0.50F, -0.50F, std::not_equal_to()));
    assert(compare_op(+0.50F, +0.50F, std::not_equal_to()));

    // operator<
    assert(compare_op(+0.00F, +0.00F, std::less()));
    assert(compare_op(+0.50F, +0.00F, std::less()));
    assert(compare_op(+0.50F, -0.50F, std::less()));
    assert(compare_op(+0.50F, +0.50F, std::less()));

    // operator<=
    assert(compare_op(+0.00F, +0.00F, std::less_equal()));
    assert(compare_op(+0.50F, +0.00F, std::less_equal()));
    assert(compare_op(+0.50F, -0.50F, std::less_equal()));
    assert(compare_op(+0.50F, +0.50F, std::less_equal()));

    // operator>
    assert(compare_op(+0.00F, +0.00F, std::greater()));
    assert(compare_op(+0.50F, +0.00F, std::greater()));
    assert(compare_op(+0.50F, -0.50F, std::greater()));
    assert(compare_op(+0.50F, +0.50F, std::greater()));

    // operator>=
    assert(compare_op(+0.00F, +0.00F, std::greater_equal()));
    assert(compare_op(+0.50F, +0.00F, std::greater_equal()));
    assert(compare_op(+0.50F, -0.50F, std::greater_equal()));
    assert(compare_op(+0.50F, +0.50F, std::greater_equal()));

    return true;
}

[[nodiscard]] static auto test_q7_t_add() -> bool
{
    using fxp_t          = neo::fft::q7_t;
    auto const tolerance = 0.01F;

    {
        // empty
        auto lhs = std::vector<fxp_t>();
        auto rhs = std::vector<fxp_t>();
        auto out = std::vector<fxp_t>();
        neo::fft::add(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});
    }

    {
        // scalar
        auto lhs = std::vector<fxp_t>{fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}};
        auto rhs = std::vector<fxp_t>{fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}};
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::fft::add(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.5F + 0.25F, tolerance); };
        assert(std::all_of(out.begin(), out.end(), eq));
    }

    {
        // vectorized
        auto lhs = std::vector<fxp_t>(1029, fxp_t{0.125F});
        auto rhs = std::vector<fxp_t>(1029, fxp_t{0.25F});
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::fft::add(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.125F + 0.25F, tolerance); };
        assert(std::all_of(out.begin(), out.end(), eq));
    }

    return true;
}

[[nodiscard]] static auto test_q15_t_add() -> bool
{
    using fxp_t          = neo::fft::q15_t;
    auto const tolerance = 0.0001F;

    {
        // empty
        auto lhs = std::vector<fxp_t>();
        auto rhs = std::vector<fxp_t>();
        auto out = std::vector<fxp_t>();
        neo::fft::add(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});
    }

    {
        // scalar
        auto lhs = std::vector<fxp_t>{fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}};
        auto rhs = std::vector<fxp_t>{fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}};
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::fft::add(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.5F + 0.25F, tolerance); };
        assert(std::all_of(out.begin(), out.end(), eq));
    }

    {
        // vectorized
        auto lhs = std::vector<fxp_t>(1029, fxp_t{0.125F});
        auto rhs = std::vector<fxp_t>(1029, fxp_t{0.25F});
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::fft::add(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.125F + 0.25F, tolerance); };
        assert(std::all_of(out.begin(), out.end(), eq));
    }

    return true;
}

[[nodiscard]] static auto test_q7_t_subtract() -> bool
{
    using fxp_t          = neo::fft::q7_t;
    auto const tolerance = 0.01F;

    {
        // empty
        auto lhs = std::vector<fxp_t>();
        auto rhs = std::vector<fxp_t>();
        auto out = std::vector<fxp_t>();
        neo::fft::subtract(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});
    }

    {
        // scalar
        auto lhs = std::vector<fxp_t>{fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}};
        auto rhs = std::vector<fxp_t>{fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}};
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::fft::subtract(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.5F - 0.25F, tolerance); };
        assert(std::all_of(out.begin(), out.end(), eq));
    }

    {
        // vectorized
        auto lhs = std::vector<fxp_t>(1029, fxp_t{0.125F});
        auto rhs = std::vector<fxp_t>(1029, fxp_t{0.25F});
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::fft::subtract(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.125F - 0.25F, tolerance); };
        assert(std::all_of(out.begin(), out.end(), eq));
    }

    return true;
}

[[nodiscard]] static auto test_q15_t_subtract() -> bool
{
    using fxp_t          = neo::fft::q15_t;
    auto const tolerance = 0.0001F;

    {
        // empty
        auto lhs = std::vector<fxp_t>();
        auto rhs = std::vector<fxp_t>();
        auto out = std::vector<fxp_t>();
        neo::fft::subtract(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});
    }

    {
        // scalar
        auto lhs = std::vector<fxp_t>{fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}};
        auto rhs = std::vector<fxp_t>{fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}};
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::fft::subtract(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.5F - 0.25F, tolerance); };
        assert(std::all_of(out.begin(), out.end(), eq));
    }

    {
        // vectorized
        auto lhs = std::vector<fxp_t>(1029, fxp_t{0.125F});
        auto rhs = std::vector<fxp_t>(1029, fxp_t{0.25F});
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::fft::subtract(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.125F - 0.25F, tolerance); };
        assert(std::all_of(out.begin(), out.end(), eq));
    }

    return true;
}

[[nodiscard]] static auto test_q7_t_multiply() -> bool
{
    using fxp_t          = neo::fft::q7_t;
    auto const tolerance = 0.01F;

    {
        // empty
        auto lhs = std::vector<fxp_t>();
        auto rhs = std::vector<fxp_t>();
        auto out = std::vector<fxp_t>();
        neo::fft::multiply(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});
    }

    {
        // scalar
        auto lhs = std::vector<fxp_t>{fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}};
        auto rhs = std::vector<fxp_t>{fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}};
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::fft::multiply(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.5F * 0.25F, tolerance); };
        assert(std::all_of(out.begin(), out.end(), eq));
    }

    {
        // vectorized
        auto lhs = std::vector<fxp_t>(1029, fxp_t{0.125F});
        auto rhs = std::vector<fxp_t>(1029, fxp_t{0.25F});
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::fft::multiply(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.125F * 0.25F, tolerance); };
        assert(std::all_of(out.begin(), out.end(), eq));
    }

    return true;
}

[[nodiscard]] static auto test_q15_t_multiply() -> bool
{
    using fxp_t          = neo::fft::q15_t;
    auto const tolerance = 0.0001F;

    {
        // empty
        auto lhs = std::vector<fxp_t>();
        auto rhs = std::vector<fxp_t>();
        auto out = std::vector<fxp_t>();
        neo::fft::multiply(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});
    }

    {
        // scalar
        auto lhs = std::vector<fxp_t>{fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}};
        auto rhs = std::vector<fxp_t>{fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}};
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::fft::multiply(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.5F * 0.25F, tolerance); };
        assert(std::all_of(out.begin(), out.end(), eq));
    }

    {
        // vectorized
        auto lhs = std::vector<fxp_t>(1029, fxp_t{0.125F});
        auto rhs = std::vector<fxp_t>(1029, fxp_t{0.25F});
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::fft::multiply(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.125F * 0.25F, tolerance); };
        assert(std::all_of(out.begin(), out.end(), eq));
    }

    return true;
}

auto main() -> int
{
    assert(test_q7_t_conversion());
    assert(test_q7_t_unary_ops());
    assert(test_q7_t_binary_ops());
    assert(test_q7_t_comparison());
    assert(test_q7_t_add());
    assert(test_q7_t_subtract());
    assert(test_q7_t_multiply());

    assert(test_q15_t_conversion());
    assert(test_q15_t_unary_ops());
    assert(test_q15_t_binary_ops());
    assert(test_q15_t_comparison());
    assert(test_q15_t_add());
    assert(test_q15_t_subtract());
    assert(test_q15_t_multiply());

    return EXIT_SUCCESS;
}
