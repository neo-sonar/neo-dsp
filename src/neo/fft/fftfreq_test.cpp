#include "fftfreq.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEMPLATE_TEST_CASE("neo/fft: fftfreq", "", float, double)
{
    using Float = TestType;

    auto const windowSize = GENERATE(128, 256, 512, 1024);
    auto const sampleRate = GENERATE(44'100.0, 48'000.0, 88'200.0, 96'000.0);

    REQUIRE(neo::fftfreq<Float>(windowSize, 0, sampleRate) == Catch::Approx(0.0));
    REQUIRE(neo::fftfreq<Float>(windowSize, windowSize / 2, sampleRate) == Catch::Approx(sampleRate / 2.0));
}
