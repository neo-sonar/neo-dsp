#include "dft.hpp"

#include <neo/fft/algorithm/allclose.hpp>
#include <neo/fft/algorithm/scale.hpp>
#include <neo/fft/testing/testing.hpp>

#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <complex>

namespace fft = neo::fft;

TEMPLATE_TEST_CASE("neo/fft/transform/dft: dft", "", float, double)
{
    using Float   = TestType;
    using Complex = std::complex<Float>;

    auto const size = GENERATE(as<std::size_t>{}, 4, 16, 21);

    auto const original = fft::generate_noise_signal<Complex>(size, Catch::getSeed());

    auto inBuf  = original;
    auto outBuf = KokkosEx::mdarray<Complex, Kokkos::dextents<size_t, 1>>{inBuf.size()};

    auto const in  = Kokkos::mdspan{inBuf.data(), Kokkos::extents{inBuf.size()}};
    auto const out = Kokkos::mdspan{outBuf.data(), Kokkos::extents{outBuf.size()}};

    neo::fft::dft(in, out, fft::direction::forward);
    neo::fft::dft(out, in, fft::direction::backward);

    fft::scale(Float(1) / static_cast<Float>(size), in);
    REQUIRE(fft::allclose(Kokkos::mdspan{original.data(), Kokkos::extents{original.size()}}, in));
}
