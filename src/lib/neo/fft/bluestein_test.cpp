#include "bluestein.hpp"

#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

TEMPLATE_TEST_CASE("neo/fft: bluestein_plan", "", std::complex<float>, std::complex<double>)
{
    using Complex = TestType;
    using Float   = typename Complex::value_type;
    using Plan    = neo::fft::bluestein_plan<Complex>;

    auto const max_size = sizeof(Float) == 8 ? 512 : 128;
    auto const size     = GENERATE(range(std::size_t(2), std::size_t(max_size)));
    CAPTURE(size);

    auto plan = Plan{size};
    REQUIRE(plan.size() == size);

    SECTION("identity")
    {
        auto x_buf = stdex::mdarray<Complex, stdex::dextents<std::size_t, 1>>{size};
        auto x     = x_buf.to_mdspan();
        x[0]       = Float(1);

        plan(x, neo::fft::direction::forward);

        for (auto i{0U}; i < size; ++i) {
            CAPTURE(i);
            REQUIRE(x[i].real() == Catch::Approx(Float(1)));
        }

        plan(x, neo::fft::direction::backward);
        REQUIRE(x[0].real() == Catch::Approx(Float(1) * Float(size)));
    }

    SECTION("random")
    {
        auto const signal = neo::generate_noise_signal<Complex>(size, Catch::getSeed());
        auto const sig    = signal.to_mdspan();

        auto x_buf = signal;
        auto x     = x_buf.to_mdspan();

        plan(x, neo::fft::direction::forward);
        plan(x, neo::fft::direction::backward);

        for (auto i{0U}; i < x.extent(0); ++i) {
            CAPTURE(i);

            auto const n     = static_cast<double>(size);
            auto const scale = neo::ilog2(size) * 4;
            REQUIRE(x[i].real() == Catch::Approx(sig[i].real() * n).scale(scale));
            REQUIRE(x[i].imag() == Catch::Approx(sig[i].imag() * n).scale(scale));
        }
    }
}
