#include "mel.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEMPLATE_TEST_CASE("neo/unit: mel_to_hertz(hertz_to_mel)", "", float, double)
{
    using Float = TestType;

    static constexpr auto const roundtrip = [](Float hertz) {
        auto const mel = neo::hertz_to_mel(hertz);
        return neo::mel_to_hertz(mel);
    };

    static constexpr auto const margin = [] {
        if constexpr (std::same_as<Float, float>) {
            return 0.0005;
        } else {
            return 0.0000001;
        }
    }();

    REQUIRE_THAT(roundtrip(Float(55)), Catch::Matchers::WithinAbs(55.0, margin));
    REQUIRE_THAT(roundtrip(Float(110)), Catch::Matchers::WithinAbs(110.0, margin));
    REQUIRE_THAT(roundtrip(Float(220)), Catch::Matchers::WithinAbs(220.0, margin));
    REQUIRE_THAT(roundtrip(Float(440)), Catch::Matchers::WithinAbs(440.0, margin));
    REQUIRE_THAT(roundtrip(Float(880)), Catch::Matchers::WithinAbs(880.0, margin));
}
