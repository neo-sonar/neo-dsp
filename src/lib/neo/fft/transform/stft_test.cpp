#include "stft.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/fft/transform: stft", "", float, double)
{
    using Float = TestType;

    SECTION("no remainder")
    {
        auto signal   = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{1, 2048};
        auto segments = neo::fft::stft(
            signal.to_mdspan(),
            neo::fft::stft_options{
                .segment_length = 256,
                .overlap_length = 0,
                .transform_size = 256,
            }
        );
        REQUIRE(segments.extent(0) == 8);
        REQUIRE(segments.extent(1) == 129);
    }

    SECTION("with remainder")
    {
        auto signal   = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{1, 2040};
        auto segments = neo::fft::stft(
            signal.to_mdspan(),
            neo::fft::stft_options{
                .segment_length = 256,
                .overlap_length = 0,
                .transform_size = 256,
            }
        );
        REQUIRE(segments.extent(0) == 8);
        REQUIRE(segments.extent(1) == 129);
    }
}
