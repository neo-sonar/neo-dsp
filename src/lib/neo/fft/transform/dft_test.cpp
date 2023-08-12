#include "dft.hpp"

#include <neo/algorithm/allclose.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/math/complex.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEMPLATE_TEST_CASE("neo/fft/transform: dft", "", float, double, long double)
{
    using Float   = TestType;
    using Complex = std::complex<Float>;

    auto const size = GENERATE(as<std::size_t>{}, 4, 16, 21);

    auto const original = neo::generate_noise_signal<Complex>(size, Catch::getSeed());

    auto inBuf  = original;
    auto outBuf = KokkosEx::mdarray<Complex, Kokkos::dextents<size_t, 1>>{inBuf.size()};

    auto const in  = Kokkos::mdspan{inBuf.data(), Kokkos::extents{inBuf.size()}};
    auto const out = Kokkos::mdspan{outBuf.data(), Kokkos::extents{outBuf.size()}};

    neo::fft::dft(in, out, neo::fft::direction::forward);
    neo::fft::dft(out, in, neo::fft::direction::backward);

    neo::scale(Float(1) / static_cast<Float>(size), in);
    REQUIRE(neo::allclose(Kokkos::mdspan{original.data(), Kokkos::extents{original.size()}}, in));
}
