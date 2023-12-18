#include "dft.hpp"

#include <neo/algorithm/allclose.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/complex.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

TEMPLATE_TEST_CASE("neo/fft: dft", "", float, double)
{
    using Float   = TestType;
    using Complex = std::complex<Float>;

    auto const size = GENERATE(as<std::size_t>{}, 4, 16, 21);

    auto const original = neo::generate_noise_signal<Complex>(size, Catch::getSeed());

    auto in_buf  = original;
    auto out_buf = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{in_buf.size()};

    auto const in  = in_buf.to_mdspan();
    auto const out = out_buf.to_mdspan();

    neo::fft::dft(in, out, neo::fft::direction::forward);
    neo::fft::dft(out, in, neo::fft::direction::backward);

    neo::scale(Float(1) / static_cast<Float>(size), in);
    REQUIRE(neo::allclose(original.to_mdspan(), in));
}
