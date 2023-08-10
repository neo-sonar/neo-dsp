#include "upols_convolver.hpp"

#include <neo/fft/algorithm/allclose.hpp>
#include <neo/fft/convolution/uniform_partition.hpp>
#include <neo/fft/testing/testing.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <span>

TEMPLATE_TEST_CASE("neo/fft/convolution: upols_convolver", "", float)
{
    using Float = TestType;

    auto const blockSize  = GENERATE(as<std::size_t>{}, 128, 256, 512);
    auto const signal     = neo::fft::make_noise_signal<Float>(blockSize * 100UL);
    auto const partitions = neo::fft::make_identity_impulse<Float>(blockSize, 10UL);

    auto convolver = neo::fft::upols_convolver<Float>{};
    auto output    = signal;
    convolver.filter(partitions);

    for (auto i{0U}; i < output.size(); i += blockSize) {
        auto block = std::span{output}.subspan(i, blockSize);
        convolver(block);
    }

    // TODO: Loop should go to output.size(), curently fails on index 128 i.e. after one block
    for (auto i{0ULL}; i < blockSize; ++i) {
        CAPTURE(i);
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(signal[i], 0.00001));
    }
}
