#include "uniform_partitioned_convolver.hpp"

#include <neo/algorithm/allclose.hpp>
#include <neo/fft/convolution/uniform_partition.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <span>

TEMPLATE_PRODUCT_TEST_CASE(
    "neo/fft/convolution: convolver",
    "",
    (neo::fft::upola_convolver, neo::fft::upols_convolver),
    (float, double, long double)
)
{
    using Convolver = TestType;
    using Float     = typename Convolver::value_type;

    auto block_size = GENERATE(as<std::size_t>{}, 128, 256, 512, 1024);

    auto const filter = [=] {
        auto const impulse = [block_size] {
            auto matrix  = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{1, block_size * 3};
            matrix(0, 0) = Float(1);
            return matrix;
        }();

        auto multi_channel = neo::fft::uniform_partition(impulse.to_mdspan(), block_size);
        auto channel_0     = stdex::submdspan(multi_channel.to_mdspan(), 0, stdex::full_extent, stdex::full_extent);
        REQUIRE(multi_channel.extent(0) == 1);

        auto mono = stdex::mdarray<std::complex<Float>, stdex::dextents<size_t, 2>>{
            multi_channel.extent(1),
            multi_channel.extent(2),
        };

        neo::scale(Float(block_size * 2), channel_0);
        neo::copy(channel_0, mono.to_mdspan());

        for (auto i{0ULL}; i < mono.extent(1); ++i) {
            CAPTURE(i);

            auto const coeff = mono(0, i);
            REQUIRE_THAT(coeff.real(), Catch::Matchers::WithinAbs(1.0, 0.00001));
            REQUIRE_THAT(coeff.imag(), Catch::Matchers::WithinAbs(0.0, 0.00001));
        }

        for (auto p{1ULL}; p < mono.extent(0); ++p) {
            CAPTURE(p);

            for (auto bin{0ULL}; bin < mono.extent(1); ++bin) {
                CAPTURE(bin);

                auto const coeff = mono(p, bin);
                REQUIRE_THAT(coeff.real(), Catch::Matchers::WithinAbs(0.0, 0.00001));
                REQUIRE_THAT(coeff.imag(), Catch::Matchers::WithinAbs(0.0, 0.00001));
            }
        }

        return mono;
    }();

    REQUIRE(filter.extent(0) == 3);
    REQUIRE(filter.extent(1) == block_size + 1);

    auto const signal = neo::generate_noise_signal<Float>(block_size * 20UL, Catch::getSeed());
    auto output       = signal;

    auto convolver = Convolver{};
    convolver.filter(filter.to_mdspan());

    for (auto i{0U}; i < output.size(); i += block_size) {
        auto const io_block = stdex::submdspan(output.to_mdspan(), std::tuple{i, i + block_size});
        convolver(io_block);
    }

    REQUIRE(neo::allclose(output.to_mdspan(), signal.to_mdspan()));
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
