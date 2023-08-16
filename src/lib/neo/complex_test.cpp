#include "complex.hpp"

#include <neo/fixed_point.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("neo/complex: is_complex")
{
    STATIC_REQUIRE(neo::is_complex<std::complex<float>>);
    STATIC_REQUIRE(neo::is_complex<std::complex<double>>);
    STATIC_REQUIRE(neo::is_complex<std::complex<long double>>);

    STATIC_REQUIRE(neo::is_complex<neo::complex64>);
    STATIC_REQUIRE(neo::is_complex<neo::complex128>);

    STATIC_REQUIRE(neo::is_complex<neo::scalar_complex<float>>);
    STATIC_REQUIRE(neo::is_complex<neo::scalar_complex<double>>);
    STATIC_REQUIRE(neo::is_complex<neo::scalar_complex<long double>>);

    STATIC_REQUIRE(not neo::is_complex<float>);
    STATIC_REQUIRE(not neo::is_complex<double>);
    STATIC_REQUIRE(not neo::is_complex<long double>);

    STATIC_REQUIRE(not neo::is_complex<signed char>);
    STATIC_REQUIRE(not neo::is_complex<signed short>);
    STATIC_REQUIRE(not neo::is_complex<signed int>);
    STATIC_REQUIRE(not neo::is_complex<signed long>);
    STATIC_REQUIRE(not neo::is_complex<signed long long>);

    STATIC_REQUIRE(not neo::is_complex<unsigned char>);
    STATIC_REQUIRE(not neo::is_complex<unsigned short>);
    STATIC_REQUIRE(not neo::is_complex<unsigned int>);
    STATIC_REQUIRE(not neo::is_complex<unsigned long>);
    STATIC_REQUIRE(not neo::is_complex<unsigned long long>);
}

TEST_CASE("neo/complex: complex")
{
    STATIC_REQUIRE(neo::complex<std::complex<float>>);
    STATIC_REQUIRE(neo::complex<std::complex<double>>);
    STATIC_REQUIRE(neo::complex<std::complex<long double>>);

    STATIC_REQUIRE(neo::complex<neo::complex64>);
    STATIC_REQUIRE(neo::complex<neo::complex128>);

    STATIC_REQUIRE(neo::complex<neo::scalar_complex<float>>);
    STATIC_REQUIRE(neo::complex<neo::scalar_complex<double>>);
    STATIC_REQUIRE(neo::complex<neo::scalar_complex<long double>>);

    STATIC_REQUIRE(not neo::complex<float>);
    STATIC_REQUIRE(not neo::complex<double>);
    STATIC_REQUIRE(not neo::complex<long double>);

    STATIC_REQUIRE(not neo::complex<signed char>);
    STATIC_REQUIRE(not neo::complex<signed short>);
    STATIC_REQUIRE(not neo::complex<signed int>);
    STATIC_REQUIRE(not neo::complex<signed long>);
    STATIC_REQUIRE(not neo::complex<signed long long>);

    STATIC_REQUIRE(not neo::complex<unsigned char>);
    STATIC_REQUIRE(not neo::complex<unsigned short>);
    STATIC_REQUIRE(not neo::complex<unsigned int>);
    STATIC_REQUIRE(not neo::complex<unsigned long>);
    STATIC_REQUIRE(not neo::complex<unsigned long long>);
}

TEST_CASE("neo/complex: float_or_complex")
{
    STATIC_REQUIRE(neo::float_or_complex<float>);
    STATIC_REQUIRE(neo::float_or_complex<double>);
    STATIC_REQUIRE(neo::float_or_complex<long double>);

    STATIC_REQUIRE(neo::float_or_complex<std::complex<float>>);
    STATIC_REQUIRE(neo::float_or_complex<std::complex<double>>);
    STATIC_REQUIRE(neo::float_or_complex<std::complex<long double>>);

    STATIC_REQUIRE(neo::float_or_complex<neo::complex64>);
    STATIC_REQUIRE(neo::float_or_complex<neo::complex128>);

    STATIC_REQUIRE(neo::float_or_complex<neo::scalar_complex<float>>);
    STATIC_REQUIRE(neo::float_or_complex<neo::scalar_complex<double>>);
    STATIC_REQUIRE(neo::float_or_complex<neo::scalar_complex<long double>>);

    STATIC_REQUIRE(not neo::float_or_complex<signed char>);
    STATIC_REQUIRE(not neo::float_or_complex<signed short>);
    STATIC_REQUIRE(not neo::float_or_complex<signed int>);
    STATIC_REQUIRE(not neo::float_or_complex<signed long>);
    STATIC_REQUIRE(not neo::float_or_complex<signed long long>);

    STATIC_REQUIRE(not neo::float_or_complex<unsigned char>);
    STATIC_REQUIRE(not neo::float_or_complex<unsigned short>);
    STATIC_REQUIRE(not neo::float_or_complex<unsigned int>);
    STATIC_REQUIRE(not neo::float_or_complex<unsigned long>);
    STATIC_REQUIRE(not neo::float_or_complex<unsigned long long>);
}

TEST_CASE("neo/complex: real_or_complex_value_t")
{
    using neo::real_or_complex_value_t;

    STATIC_REQUIRE(std::same_as<real_or_complex_value_t<float>, float>);
    STATIC_REQUIRE(std::same_as<real_or_complex_value_t<double>, double>);
    STATIC_REQUIRE(std::same_as<real_or_complex_value_t<long double>, long double>);

    STATIC_REQUIRE(std::same_as<real_or_complex_value_t<std::complex<float>>, float>);
    STATIC_REQUIRE(std::same_as<real_or_complex_value_t<std::complex<double>>, double>);
    STATIC_REQUIRE(std::same_as<real_or_complex_value_t<std::complex<long double>>, long double>);

    STATIC_REQUIRE(std::same_as<real_or_complex_value_t<neo::complex64>, float>);
    STATIC_REQUIRE(std::same_as<real_or_complex_value_t<neo::complex128>, double>);

    STATIC_REQUIRE(std::same_as<real_or_complex_value_t<neo::scalar_complex<float>>, float>);
    STATIC_REQUIRE(std::same_as<real_or_complex_value_t<neo::scalar_complex<double>>, double>);
    STATIC_REQUIRE(std::same_as<real_or_complex_value_t<neo::scalar_complex<long double>>, long double>);
}

TEMPLATE_TEST_CASE("neo/complex: scalar_complex", "", float, double, long double)
{
    using Float   = TestType;
    using Complex = neo::scalar_complex<Float>;

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
