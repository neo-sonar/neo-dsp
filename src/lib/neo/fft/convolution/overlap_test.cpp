#include "overlap_add.hpp"
#include "overlap_save.hpp"

#include <neo/algorithm/rms_error.hpp>
#include <neo/fft/convolution/uniform_partition.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <tuple>

using namespace neo::fft;

TEMPLATE_PRODUCT_TEST_CASE("neo/fft/convolution:", "", (overlap_add, overlap_save), (float, double, long double))
{
    using Overlap = TestType;
    using Float   = typename Overlap::real_type;

    auto const blockSize = GENERATE(as<std::size_t>{}, 128, 256, 512);
    auto const signal    = neo::generate_noise_signal<Float>(blockSize * 15UL, Catch::getSeed());

    auto overlap = Overlap{blockSize, blockSize};

    REQUIRE(overlap.block_size() == blockSize);
    REQUIRE(overlap.filter_size() == blockSize);
    REQUIRE(overlap.transform_size() == blockSize * 2UL);

    auto output = signal;
    auto blocks = Kokkos::mdspan{output.data(), Kokkos::extents{output.size()}};

    for (auto i{0U}; i < output.size(); i += blockSize) {
        auto block = KokkosEx::submdspan(blocks, std::tuple{i, i + blockSize});
        overlap(block, [=](neo::inout_vector auto io) {
            REQUIRE(io.extent(0) == overlap.transform_size() / 2UL + 1UL);
        });
    }

    auto const sig = signal.to_mdspan();
    auto const out = output.to_mdspan();

    auto const error = neo::rms_error(sig, out);
    REQUIRE_THAT(error, Catch::Matchers::WithinAbs(0.0, 0.00001));

    for (auto i{0ULL}; i < output.size(); ++i) {
        CAPTURE(i);
        REQUIRE_THAT(out[i], Catch::Matchers::WithinAbs(sig[i], 0.00001));
    }
}
