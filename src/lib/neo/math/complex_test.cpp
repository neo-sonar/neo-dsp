#include "complex.hpp"

#include <catch2/catch_test_macros.hpp>

namespace {
namespace ns {
struct complex
{
    using value_type = short;
};
}  // namespace ns
}  // namespace

template<>
inline constexpr auto const neo::is_complex<ns::complex> = true;

TEST_CASE("neo/math: is_complex")
{
    STATIC_REQUIRE(neo::is_complex<std::complex<float>>);
    STATIC_REQUIRE(neo::is_complex<std::complex<double>>);
    STATIC_REQUIRE(neo::is_complex<std::complex<long double>>);
    STATIC_REQUIRE(neo::is_complex<ns::complex>);

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

TEST_CASE("neo/math: complex")
{
    STATIC_REQUIRE(neo::complex<std::complex<float>>);
    STATIC_REQUIRE(neo::complex<std::complex<double>>);
    STATIC_REQUIRE(neo::complex<std::complex<long double>>);
    STATIC_REQUIRE(neo::complex<ns::complex>);

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

TEST_CASE("neo/math: float_or_complex")
{
    STATIC_REQUIRE(neo::float_or_complex<float>);
    STATIC_REQUIRE(neo::float_or_complex<double>);
    STATIC_REQUIRE(neo::float_or_complex<long double>);

    STATIC_REQUIRE(neo::float_or_complex<std::complex<float>>);
    STATIC_REQUIRE(neo::float_or_complex<std::complex<double>>);
    STATIC_REQUIRE(neo::float_or_complex<std::complex<long double>>);

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

TEST_CASE("neo/math: real_or_complex_value_t")
{
    using neo::real_or_complex_value_t;

    STATIC_REQUIRE(std::same_as<real_or_complex_value_t<float>, float>);
    STATIC_REQUIRE(std::same_as<real_or_complex_value_t<double>, double>);
    STATIC_REQUIRE(std::same_as<real_or_complex_value_t<long double>, long double>);

    STATIC_REQUIRE(std::same_as<real_or_complex_value_t<std::complex<float>>, float>);
    STATIC_REQUIRE(std::same_as<real_or_complex_value_t<std::complex<double>>, double>);
    STATIC_REQUIRE(std::same_as<real_or_complex_value_t<std::complex<long double>>, long double>);
    STATIC_REQUIRE(std::same_as<real_or_complex_value_t<ns::complex>, short>);
}
