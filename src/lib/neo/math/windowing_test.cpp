#include "windowing.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEMPLATE_PRODUCT_TEST_CASE(
    "neo/math: windowing",
    "",
    (neo::hann_window, neo::hamming_window),
    (float, double, long double)
)
{
    using Window = TestType;
    using Float  = typename Window::real_type;

    auto const size = GENERATE(as<std::size_t>{}, 128, 256, 512, 1024);

    SECTION("edges")
    {
        auto const window = Window{};
        // REQUIRE(window(0, size) == Catch::Approx(0.0));
        // REQUIRE(window(size - 1, size) == Catch::Approx(0.0));
        REQUIRE_THAT(window(size / 2 - 1, size), Catch::Matchers::WithinAbs(1.0, 0.01));
    }

    SECTION("generate_window")
    {
        auto const window = neo::generate_window<Float, Window>(size);
        STATIC_REQUIRE(decltype(window)::rank() == 1);
        // REQUIRE(window(0) == Catch::Approx(0.0));
        // REQUIRE(window(size - 1) == Catch::Approx(0.0));
        REQUIRE_THAT(window(size / 2 - 1), Catch::Matchers::WithinAbs(1.0, 0.01));
    }
}
