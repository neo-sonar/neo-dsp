#include "rfftfreq.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEMPLATE_TEST_CASE("neo/fft: rfftfreq", "", float, double)
{
    using Float = TestType;

    auto const window_size     = GENERATE(128, 256, 512, 1024);
    auto const sample_rate     = GENERATE(44'100.0, 48'000.0, 88'200.0, 96'000.0);
    auto const inv_sample_rate = 1.0 / sample_rate;

    REQUIRE(neo::rfftfreq<Float>(window_size, 0, inv_sample_rate) == Catch::Approx(0.0));
    REQUIRE(neo::rfftfreq<Float>(window_size, window_size / 2, inv_sample_rate) == Catch::Approx(sample_rate / 2.0));

    auto freqs = std::array<Float, 2>{};
    neo::rfftfreq(stdex::mdspan{freqs.data(), stdex::extents{freqs.size()}}, inv_sample_rate);
    REQUIRE(freqs[0] == Catch::Approx(0.0));
    REQUIRE(freqs[1] == Catch::Approx(sample_rate / 2.0));  // TODO: Doesn't match scipy output. Missing negative freqs
}
