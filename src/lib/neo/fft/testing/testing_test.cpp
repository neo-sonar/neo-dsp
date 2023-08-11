#include "testing.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/fft/testing: load_test_data_file", "", float, double)
{
    using Float = TestType;

    REQUIRE_FALSE(neo::fft::load_test_data_file<Float>("./does/not/exist").has_value());
    REQUIRE_FALSE(neo::fft::load_test_data_file<Float>("./src").has_value());  // not regular file
}

TEMPLATE_TEST_CASE("neo/fft/testing: load_test_data", "", float, double)
{
    using Float = TestType;
    using neo::fft::load_test_data;

    REQUIRE_FALSE(load_test_data<Float>({}).has_value());
    REQUIRE_FALSE(load_test_data<Float>({"./test_data/c2c_8_input.csv", {}}).has_value());

    REQUIRE(load_test_data<Float>({"./test_data/c2c_8_input.csv", "./test_data/c2c_8_output.csv"}).has_value());
}
