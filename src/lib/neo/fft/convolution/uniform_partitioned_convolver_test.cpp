#include "sparse_upols_convolver.hpp"
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

namespace {
template<typename T>
static constexpr auto is_sparse_convolver = false;

template<typename Float>
static constexpr auto is_sparse_convolver<neo::fft::sparse_upols_convolver<Float>> = true;
}  // namespace

static_assert(not is_sparse_convolver<neo::fft::upola_convolver<float>>);
static_assert(not is_sparse_convolver<neo::fft::upols_convolver<float>>);
static_assert(is_sparse_convolver<neo::fft::sparse_upols_convolver<float>>);

TEMPLATE_PRODUCT_TEST_CASE(
    "neo/fft/convolution: convolver",
    "",
    (neo::fft::upola_convolver, neo::fft::upols_convolver, neo::fft::sparse_upols_convolver),
    (float, double, long double)
)
{
    using Convolver = TestType;
    using Float     = typename Convolver::value_type;

    auto const block_size = GENERATE(as<std::size_t>{}, 128, 256, 512, 1024);

    auto const filter = neo::generate_identity_impulse<Float>(block_size, 3);
    REQUIRE(filter.extent(0) == 3);
    REQUIRE(filter.extent(1) == block_size + 1);

    auto const signal = neo::generate_noise_signal<Float>(block_size * 20UL, Catch::getSeed());
    auto output       = signal;

    auto convolver = Convolver{};
    if constexpr (is_sparse_convolver<Convolver>) {
        convolver.filter(filter.to_mdspan(), [](auto, auto, auto) { return true; });
    } else {
        convolver.filter(filter.to_mdspan());
    }

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
