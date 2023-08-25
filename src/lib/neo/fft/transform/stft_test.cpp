#include "stft.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEST_CASE("neo/fft/transform: detail::num_sftf_frames")
{
    REQUIRE(neo::fft::detail::num_sftf_frames(1024, 128, 0) == 8);
    REQUIRE(neo::fft::detail::num_sftf_frames(1024, 256, 0) == 4);
    REQUIRE(neo::fft::detail::num_sftf_frames(1024, 256, 128) == 8);
}

TEMPLATE_TEST_CASE("neo/fft/transform: stft", "", float, double)
{
    using Float = TestType;

    auto const size   = GENERATE(as<std::size_t>{}, 2040, 2048);
    auto const signal = stdex::mdarray<Float, stdex::dextents<std::size_t, 2>>{1, size};

    auto const no_overlap_options = neo::fft::stft_options{
        .frame_length   = 256,
        .overlap_length = 0,
        .transform_size = 256,
    };

    auto no_overlap = neo::fft::stft(signal.to_mdspan(), no_overlap_options);
    REQUIRE(no_overlap.extent(0) == 8);
    REQUIRE(no_overlap.extent(1) == 129);

    auto half_overlap = neo::fft::stft(signal.to_mdspan(), 256);
    REQUIRE(half_overlap.extent(0) == 16);
    REQUIRE(half_overlap.extent(1) == 129);
}
