#include "neo/convolution/math/fixed_point.hpp"

#undef NDEBUG
#include <cassert>
#include <cstdio>
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

[[nodiscard]] static auto test_q7_t_arithmetic() -> bool
{
    using fxp_t          = neo::fft::q7_t;
    auto const tolerance = 0.01F;

    // operator+
    assert(approx_equal(to_float(fxp_t{0.00F} + fxp_t{0.5F}), 0.00F + 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.12F} + fxp_t{0.5F}), 0.12F + 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.25F} + fxp_t{0.5F}), 0.25F + 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.33F} + fxp_t{0.5F}), 0.33F + 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.40F} + fxp_t{0.5F}), 0.40F + 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.45F} + fxp_t{0.5F}), 0.45F + 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.49F} + fxp_t{0.5F}), 0.49F + 0.5F, tolerance));

    // operator-
    assert(approx_equal(to_float(fxp_t{0.00F} - fxp_t{0.5F}), 0.00F - 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.12F} - fxp_t{0.5F}), 0.12F - 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.25F} - fxp_t{0.5F}), 0.25F - 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.33F} - fxp_t{0.5F}), 0.33F - 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.40F} - fxp_t{0.5F}), 0.40F - 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.45F} - fxp_t{0.5F}), 0.45F - 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.49F} - fxp_t{0.5F}), 0.49F - 0.5F, tolerance));

    // operator*
    assert(approx_equal(to_float(fxp_t{0.00F} * fxp_t{0.5F}), 0.00F * 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.12F} * fxp_t{0.5F}), 0.12F * 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.25F} * fxp_t{0.5F}), 0.25F * 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.33F} * fxp_t{0.5F}), 0.33F * 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.40F} * fxp_t{0.5F}), 0.40F * 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.45F} * fxp_t{0.5F}), 0.45F * 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.49F} * fxp_t{0.5F}), 0.49F * 0.5F, tolerance));

    return true;
}

[[nodiscard]] static auto test_q15_t_arithmetic() -> bool
{
    using fxp_t          = neo::fft::q15_t;
    auto const tolerance = 0.0001F;

    // operator+
    assert(approx_equal(to_float(fxp_t{0.00F} + fxp_t{0.5F}), 0.00F + 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.12F} + fxp_t{0.5F}), 0.12F + 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.25F} + fxp_t{0.5F}), 0.25F + 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.33F} + fxp_t{0.5F}), 0.33F + 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.40F} + fxp_t{0.5F}), 0.40F + 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.45F} + fxp_t{0.5F}), 0.45F + 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.49F} + fxp_t{0.5F}), 0.49F + 0.5F, tolerance));

    // operator-
    assert(approx_equal(to_float(fxp_t{0.00F} - fxp_t{0.5F}), 0.00F - 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.12F} - fxp_t{0.5F}), 0.12F - 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.25F} - fxp_t{0.5F}), 0.25F - 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.33F} - fxp_t{0.5F}), 0.33F - 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.40F} - fxp_t{0.5F}), 0.40F - 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.45F} - fxp_t{0.5F}), 0.45F - 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.49F} - fxp_t{0.5F}), 0.49F - 0.5F, tolerance));

    // operator*
    assert(approx_equal(to_float(fxp_t{0.00F} * fxp_t{0.5F}), 0.00F * 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.12F} * fxp_t{0.5F}), 0.12F * 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.25F} * fxp_t{0.5F}), 0.25F * 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.33F} * fxp_t{0.5F}), 0.33F * 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.40F} * fxp_t{0.5F}), 0.40F * 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.45F} * fxp_t{0.5F}), 0.45F * 0.5F, tolerance));
    assert(approx_equal(to_float(fxp_t{0.49F} * fxp_t{0.5F}), 0.49F * 0.5F, tolerance));

    return true;
}

[[nodiscard]] static auto test_q15_t_fixed_point_multiply() -> bool
{
    using fxp_t          = neo::fft::q15_t;
    auto const tolerance = 0.0001F;

    {
        // empty
        auto lhs = std::vector<fxp_t>();
        auto rhs = std::vector<fxp_t>();
        auto out = std::vector<fxp_t>();
        neo::fft::fixed_point_multiply(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});
    }

    {
        // scalar
        auto lhs = std::vector<fxp_t>{fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}, fxp_t{0.5}};
        auto rhs = std::vector<fxp_t>{fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}, fxp_t{0.25}};
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::fft::fixed_point_multiply(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.5F * 0.25F, tolerance); };
        assert(std::all_of(out.begin(), out.end(), eq));
    }

    {
        // vectorized
        auto lhs = std::vector<fxp_t>(1029, fxp_t{0.125F});
        auto rhs = std::vector<fxp_t>(1029, fxp_t{0.25F});
        auto out = std::vector<fxp_t>(lhs.size(), fxp_t{});
        neo::fft::fixed_point_multiply(std::span{std::as_const(lhs)}, std::span{std::as_const(rhs)}, std::span{out});

        auto eq = [=](auto fxp) { return approx_equal(to_float(fxp), 0.125F * 0.25F, tolerance); };
        assert(std::all_of(out.begin(), out.end(), eq));
    }

    return true;
}

auto main() -> int
{
    assert(test_q7_t_conversion());
    assert(test_q15_t_conversion());

    assert(test_q7_t_arithmetic());
    assert(test_q15_t_arithmetic());

    assert(test_q15_t_fixed_point_multiply());

    return EXIT_SUCCESS;
}
