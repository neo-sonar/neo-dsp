#include "uniform_partitioned_convolver.hpp"

#include <neo/fft/convolution/uniform_partition.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <span>

template<std::floating_point Float, typename Convolver>
[[nodiscard]] static auto test_uniform_partitioned_convolver(auto blockSize)
{
    auto const signal     = neo::generate_noise_signal<Float>(blockSize * 20UL, Catch::getSeed());
    auto const partitions = neo::generate_identity_impulse<Float>(blockSize, 10UL);

    auto convolver = Convolver{};
    auto output    = signal;
    convolver.filter(partitions.to_mdspan());

    for (auto i{0U}; i < output.size(); i += blockSize) {
        auto block = std::span{output}.subspan(i, blockSize);
        convolver(Kokkos::mdspan{block.data(), Kokkos::extents{block.size()}});
    }

    // TODO: Loop should go to output.size(), curently fails on index 128 i.e. after one block
    for (auto i{0ULL}; i < blockSize; ++i) {
        CAPTURE(i);
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(signal[i], 0.00001));
    }
}

TEMPLATE_TEST_CASE("neo/fft/convolution: upols_convolver", "", float, double)
{
    using Float    = TestType;
    auto blockSize = GENERATE(as<std::size_t>{}, 128, 256, 512);
    test_uniform_partitioned_convolver<Float, neo::fft::upols_convolver<Float>>(blockSize);
}

TEMPLATE_TEST_CASE("neo/fft/convolution: upola_convolver", "", float, double)
{
    using Float    = TestType;
    auto blockSize = GENERATE(as<std::size_t>{}, 128, 256, 512);
    test_uniform_partitioned_convolver<Float, neo::fft::upola_convolver<Float>>(blockSize);
}
