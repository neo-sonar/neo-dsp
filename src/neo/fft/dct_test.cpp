#include "dct.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEMPLATE_TEST_CASE("neo/fft: dct2_plan", "", float, double)
{
    using Float = TestType;
    using Plan  = neo::fft::dct2_plan<Float>;

    SECTION("size/order")
    {
        auto const order = GENERATE(as<std::size_t>{}, 2, 3, 4);
        auto const size  = std::size_t(1) << order;

        auto plan = Plan{order};
        REQUIRE(plan.size() == size);
        REQUIRE(plan.order() == order);
    }

    SECTION("python: scipy.fft.dct(x, type=2)")
    {
        auto plan = Plan{3};
        auto x    = std::array<Float, 8>{1, 2, 3, 4, 5, 6, 7, 8};
        plan(stdex::mdspan{x.data(), stdex::extents{x.size()}});

        REQUIRE(plan.size() == 8);
        REQUIRE(x[0] == Catch::Approx(72.0));
        REQUIRE(x[1] == Catch::Approx(-25.76929209));
        REQUIRE(x[2] == Catch::Approx(0.0));
        REQUIRE(x[3] == Catch::Approx(-2.6938192));
        REQUIRE(x[4] == Catch::Approx(0.0));
        REQUIRE(x[5] == Catch::Approx(-0.80361161));
        REQUIRE(x[6] == Catch::Approx(0.0));
        REQUIRE(x[7] == Catch::Approx(-0.20280929));
    }
}
