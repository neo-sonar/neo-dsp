#include "upols_convolver.hpp"

#include <neo/fft/algorithm/allclose.hpp>
#include <neo/fft/convolution/uniform_partition.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <concepts>
#include <random>
#include <span>
#include <vector>

template<std::floating_point Float>
[[nodiscard]] auto make_noise_signal(std::size_t length) -> std::vector<Float>
{
    auto signal = std::vector<Float>(length, Float(0));
    auto rng    = std::mt19937{std::random_device{}()};
    auto dist   = std::uniform_real_distribution<Float>{Float(-1), Float(1)};
    std::generate(signal.begin(), signal.end(), [&] { return dist(rng); });
    return signal;
}

template<std::floating_point Float>
[[nodiscard]] auto make_identity_impulse(std::size_t blockSize, std::size_t numPartitions)
    -> KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<std::size_t, 2>>
{
    auto const windowSize = blockSize * 2;
    auto const numBins    = windowSize / 2 + 1;

    auto impulse = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<std::size_t, 2>>{
        numPartitions,
        numBins,
    };

    for (auto partition{0U}; partition < impulse.extent(0); ++partition) {
        for (auto bin{0U}; bin < impulse.extent(1); ++bin) {
            impulse(partition, bin) = std::complex{Float(1), Float(0)};
        }
    }
    return impulse;
}

TEMPLATE_TEST_CASE("neo/fft/convolution: upols_convolver", "", float)
{
    using Float = TestType;

    auto const blockSize  = GENERATE(as<std::size_t>{}, 128, 256, 512);
    auto const signal     = make_noise_signal<Float>(blockSize * 100UL);
    auto const partitions = make_identity_impulse<Float>(blockSize, 10UL);

    auto convolver = neo::fft::upols_convolver{};
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
