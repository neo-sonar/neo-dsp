#include "scalar_complex.hpp"

#include <neo/fixed_point.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/complex: scalar_complex", "", float, double, long double)
{
    using Float   = TestType;
    using Complex = neo::scalar_complex<Float>;

    STATIC_REQUIRE(neo::is_complex<neo::scalar_complex<Float>>);
    STATIC_REQUIRE(neo::complex<neo::scalar_complex<Float>>);
    STATIC_REQUIRE(neo::float_or_complex<neo::scalar_complex<Float>>);
    STATIC_REQUIRE(std::same_as<neo::real_or_complex_value_t<neo::scalar_complex<Float>>, Float>);
    STATIC_REQUIRE(std::same_as<typename Complex::value_type, Float>);

    auto tc = Complex{Float(1)};
    REQUIRE(tc.real() == Catch::Approx(Float(1)));
    REQUIRE(tc.imag() == Catch::Approx(Float(0)));

    tc.real(Float(2));
    tc.imag(Float(3));
    REQUIRE(tc.real() == Catch::Approx(Float(2)));
    REQUIRE(tc.imag() == Catch::Approx(Float(3)));

    auto const sum = tc + tc;
    REQUIRE(sum.real() == Catch::Approx(Float(4)));
    REQUIRE(sum.imag() == Catch::Approx(Float(6)));

    auto const diff = tc - sum;
    REQUIRE(diff.real() == Catch::Approx(Float(-2)));
    REQUIRE(diff.imag() == Catch::Approx(Float(-3)));

    auto const product = tc * sum;
    REQUIRE(product.real() == Catch::Approx(Float(-10)));
    REQUIRE(product.imag() == Catch::Approx(Float(24)));
}

TEMPLATE_TEST_CASE("neo/complex: scalar_complex", "", neo::q7, neo::q15)
{
    using FixedPoint = TestType;
    using Complex    = neo::scalar_complex<FixedPoint>;

    STATIC_REQUIRE(neo::is_complex<neo::scalar_complex<FixedPoint>>);
    STATIC_REQUIRE(neo::complex<neo::scalar_complex<FixedPoint>>);
    STATIC_REQUIRE(neo::float_or_complex<neo::scalar_complex<FixedPoint>>);
    STATIC_REQUIRE(std::same_as<neo::real_or_complex_value_t<neo::scalar_complex<FixedPoint>>, FixedPoint>);
    STATIC_REQUIRE(std::same_as<typename Complex::value_type, FixedPoint>);

    auto tc = Complex{FixedPoint(neo::underlying_value, 10)};
    REQUIRE(tc.real().value() == 10);
    REQUIRE(tc.imag().value() == 0);

    tc.real(FixedPoint(neo::underlying_value, 20));
    tc.imag(FixedPoint(neo::underlying_value, 30));
    REQUIRE(tc.real().value() == 20);
    REQUIRE(tc.imag().value() == 30);

    auto const sum = tc + tc;
    REQUIRE(sum.real().value() == 40);
    REQUIRE(sum.imag().value() == 60);

    auto const diff = tc - sum;
    REQUIRE(diff.real().value() == -20);
    REQUIRE(diff.imag().value() == -30);
}
