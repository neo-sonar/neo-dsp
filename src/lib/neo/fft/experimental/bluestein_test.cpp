#include "bluestein.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("neo/fft: experimental::bluestein")
{
    auto const margin = 0.000001;
    auto const size   = GENERATE(as<std::size_t>{}, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
    CAPTURE(size);

    auto re = std::vector<double>(size);
    auto im = std::vector<double>(size);
    re[0]   = 1.0;

    // forward
    neo::fft::experimental::bluestein(re, im);
    for (auto i{0U}; i < size; ++i) {
        REQUIRE_THAT(re[i], Catch::Matchers::WithinAbs(1.0, margin));
        REQUIRE_THAT(std::abs(im[i]), Catch::Matchers::WithinAbs(0.0, margin));
    }

    // backward
    neo::fft::experimental::bluestein(im, re);
    REQUIRE_THAT(re[0], Catch::Matchers::WithinAbs(1.0 * double(size), margin));
    REQUIRE_THAT(std::abs(im[0]), Catch::Matchers::WithinAbs(0.0, margin));

    for (auto i{1U}; i < size; ++i) {
        REQUIRE_THAT(re[i], Catch::Matchers::WithinAbs(0.0, margin));
        REQUIRE_THAT(std::abs(im[i]), Catch::Matchers::WithinAbs(0.0, margin));
    }
}

TEMPLATE_TEST_CASE("neo/fft: experimental::bluestein_plan", "", std::complex<float>, std::complex<double>)
{
    using Complex = TestType;
    using Plan    = neo::fft::experimental::bluestein_plan<Complex>;

    auto const size = GENERATE(as<std::size_t>{}, 4, 5, 6, 7, 8);
    CAPTURE(size);

    auto const plan = Plan{size};
    REQUIRE(plan.size() == size);
}
