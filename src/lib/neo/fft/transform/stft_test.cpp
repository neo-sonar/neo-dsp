#include "stft.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/fft/transform: stft", "", float, double, long double)
{
    using Float = TestType;

    SECTION("no remainder")
    {
        auto signal = KokkosEx::mdarray<Float, Kokkos::dextents<std::size_t, 2>>{1, 2048};
        auto frames = neo::fft::stft(signal.to_mdspan(), 256);
        REQUIRE(frames.extent(0) == 8);
        REQUIRE(frames.extent(1) == 129);
    }

    SECTION("with remainder")
    {
        auto signal = KokkosEx::mdarray<Float, Kokkos::dextents<std::size_t, 2>>{1, 2040};
        auto frames = neo::fft::stft(signal.to_mdspan(), 256);
        REQUIRE(frames.extent(0) == 8);
        REQUIRE(frames.extent(1) == 129);
    }
}
