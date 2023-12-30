// SPDX-License-Identifier: MIT
#include "complex.hpp"

#include <neo/complex/scalar_complex.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("neo/complex: is_complex")
{
    STATIC_REQUIRE(neo::is_complex<std::complex<float>>);
    STATIC_REQUIRE(neo::is_complex<std::complex<double>>);
    STATIC_REQUIRE(neo::is_complex<std::complex<long double>>);

    STATIC_REQUIRE(neo::is_complex<neo::complex64>);
    STATIC_REQUIRE(neo::is_complex<neo::complex128>);

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
}
