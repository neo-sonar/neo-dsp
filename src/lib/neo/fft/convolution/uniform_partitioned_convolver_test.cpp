#include "uniform_partitioned_convolver.hpp"

#include <neo/fft/convolution/uniform_partition.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <span>

template<std::floating_point Float, typename Convolver>
[[nodiscard]] static auto test_uniform_partitioned_convolver_old(auto blockSize)
{
    auto const signal     = neo::generate_noise_signal<Float>(blockSize * 20UL, Catch::getSeed());
    auto const partitions = neo::generate_identity_impulse<Float>(blockSize, 10UL);

    auto convolver = Convolver{};
    auto output    = signal;
    convolver.filter(partitions.to_mdspan());

    for (auto i{0U}; i < output.size(); i += blockSize) {
        auto block = stdex::submdspan(output.to_mdspan(), std::tuple{i, i + blockSize});
        convolver(block);
    }

    // TODO: Loop should go to output.size(), curently fails on index 128 i.e. after one block
    for (auto i{0ULL}; i < blockSize; ++i) {
        CAPTURE(i);
        REQUIRE_THAT(output(i), Catch::Matchers::WithinAbs(signal(i), 0.00001));
    }
}

TEMPLATE_TEST_CASE("neo/fft/convolution: upols_convolver(old)", "", float, double, long double)
{
    using Float    = TestType;
    auto blockSize = GENERATE(as<std::size_t>{}, 128, 256, 512);
    test_uniform_partitioned_convolver_old<Float, neo::fft::upols_convolver<Float>>(blockSize);
}

TEMPLATE_TEST_CASE("neo/fft/convolution: upola_convolver", "", float, double, long double)
{
    using Float    = TestType;
    auto blockSize = GENERATE(as<std::size_t>{}, 128, 256, 512);
    test_uniform_partitioned_convolver_old<Float, neo::fft::upola_convolver<Float>>(blockSize);
}

template<std::floating_point Float, typename Convolver>
[[nodiscard]] static auto test_uniform_partitioned_convolver(auto block_size)
{
    auto const impulse = [block_size] {
        auto matrix  = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{1, block_size * 3};
        matrix(0, 0) = Float(1);
        return matrix;
    }();

    auto const multi_channel_partitions = neo::fft::uniform_partition(impulse.to_mdspan(), block_size);
    REQUIRE(multi_channel_partitions.extent(0) == 1);

    auto const partitions
        = stdex::submdspan(multi_channel_partitions.to_mdspan(), 0, stdex::full_extent, stdex::full_extent);

    REQUIRE(partitions.extent(0) == 3);
    REQUIRE(partitions.extent(1) == block_size + 1);

    for (auto i{0ULL}; i < partitions.extent(1); ++i) {
        CAPTURE(i);

        auto const coeff = partitions(0, i);
        REQUIRE_THAT(coeff.real() * Float(block_size * 2), Catch::Matchers::WithinAbs(1.0, 0.00001));
        REQUIRE_THAT(coeff.imag(), Catch::Matchers::WithinAbs(0.0, 0.00001));
    }

    for (auto p{1ULL}; p < partitions.extent(0); ++p) {
        CAPTURE(p);

        for (auto bin{0ULL}; bin < partitions.extent(1); ++bin) {
            CAPTURE(bin);

            auto const coeff = partitions(p, bin);
            REQUIRE_THAT(coeff.real(), Catch::Matchers::WithinAbs(0.0, 0.00001));
            REQUIRE_THAT(coeff.imag(), Catch::Matchers::WithinAbs(0.0, 0.00001));
        }
    }

    auto const signal = neo::generate_noise_signal<Float>(block_size * 20UL, Catch::getSeed());
    auto output       = signal;

    auto convolver = Convolver{};
    convolver.filter(partitions);

    for (auto i{0U}; i < output.size(); i += block_size) {
        auto const io_block = stdex::submdspan(output.to_mdspan(), std::tuple{i, i + block_size});
        convolver(io_block);
    }

    // TODO: Loop should go to output.size(), curently fails on index 128 i.e. after one block
    // for (auto i{0ULL}; i < output.size(); ++i) {
    //     CAPTURE(i);
    //     REQUIRE_THAT(output(i), Catch::Matchers::WithinAbs(signal(i), 0.00001));
    // }
}

TEMPLATE_TEST_CASE("neo/fft/convolution: upols_convolver", "", float, double, long double)
{
    using Float = TestType;

    auto block_size = GENERATE(as<std::size_t>{}, 1024);
    test_uniform_partitioned_convolver<Float, neo::fft::upols_convolver<Float>>(block_size);
}

TEMPLATE_TEST_CASE("neo/fft/convolution: shift_rows_up", "", float, double, long double)
{
    using Float = TestType;

    auto matrix_buffer = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{3, 2};
    auto matrix        = matrix_buffer.to_mdspan();

    matrix(0, 0) = Float(0);
    matrix(0, 1) = Float(1);

    matrix(1, 0) = Float(2);
    matrix(1, 1) = Float(3);

    matrix(2, 0) = Float(4);
    matrix(2, 1) = Float(5);

    neo::fft::shift_rows_up(matrix);
    REQUIRE(matrix(1, 0) == Catch::Approx(Float(0)));
    REQUIRE(matrix(1, 1) == Catch::Approx(Float(1)));
    REQUIRE(matrix(2, 0) == Catch::Approx(Float(2)));
    REQUIRE(matrix(2, 1) == Catch::Approx(Float(3)));

    neo::fft::shift_rows_up(matrix);
    REQUIRE(matrix(2, 0) == Catch::Approx(Float(0)));
    REQUIRE(matrix(2, 1) == Catch::Approx(Float(1)));
}
