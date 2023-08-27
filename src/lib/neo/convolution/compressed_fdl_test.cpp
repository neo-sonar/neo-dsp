#include "compressed_fdl.hpp"

#include <neo/algorithm/add.hpp>
#include <neo/algorithm/fill.hpp>
#include <neo/convolution/dense_convolver.hpp>
#include <neo/convolution/overlap_add.hpp>
#include <neo/convolution/uniform_partitioned_convolver.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEMPLATE_TEST_CASE("neo/convolution: compressed_fdl", "", std::int8_t, std::int16_t)
{
    using Int          = TestType;
    using IntComplex   = neo::scalar_complex<Int>;
    using FloatComplex = neo::scalar_complex<float>;
    using Fdl          = neo::fft::compressed_fdl<FloatComplex, IntComplex>;

    STATIC_REQUIRE(std::same_as<typename Fdl::value_type, FloatComplex>);
    STATIC_REQUIRE(std::same_as<typename Fdl::compressed_type, IntComplex>);

    static constexpr auto tolerance = [] {
        if constexpr (std::same_as<Int, int8_t>) {
            return 0.005;
        }
        return 0.0001;
    }();

    auto input = stdex::mdarray<FloatComplex, stdex::dextents<size_t, 1>>{8};

    input(0) = FloatComplex{+0.000F, +0.125F};
    input(1) = FloatComplex{+0.250F, +0.333F};
    input(2) = FloatComplex{+0.500F, +0.666F};
    input(3) = FloatComplex{+0.750F, +1.000F};

    input(4) = FloatComplex{-0.000F, -0.125F};
    input(5) = FloatComplex{-0.250F, -0.333F};
    input(6) = FloatComplex{-0.500F, -0.666F};
    input(7) = FloatComplex{-0.750F, -1.000F};

    auto fdl = Fdl(stdex::extents{4, 8});
    fdl(input.to_mdspan(), 0);

    auto compressed = fdl(0);
    REQUIRE(compressed.extent(0) == 8);

    REQUIRE_THAT(compressed[0].real(), Catch::Matchers::WithinAbs(+0.000, tolerance));
    REQUIRE_THAT(compressed[0].imag(), Catch::Matchers::WithinAbs(+0.125, tolerance));

    REQUIRE_THAT(compressed[1].real(), Catch::Matchers::WithinAbs(+0.250, tolerance));
    REQUIRE_THAT(compressed[1].imag(), Catch::Matchers::WithinAbs(+0.333, tolerance));

    REQUIRE_THAT(compressed[2].real(), Catch::Matchers::WithinAbs(+0.500, tolerance));
    REQUIRE_THAT(compressed[2].imag(), Catch::Matchers::WithinAbs(+0.666, tolerance));

    REQUIRE_THAT(compressed[3].real(), Catch::Matchers::WithinAbs(+0.750, tolerance));
    REQUIRE_THAT(compressed[3].imag(), Catch::Matchers::WithinAbs(+1.000, tolerance));

    REQUIRE_THAT(compressed[4].real(), Catch::Matchers::WithinAbs(-0.000, tolerance));
    REQUIRE_THAT(compressed[4].imag(), Catch::Matchers::WithinAbs(-0.125, tolerance));

    REQUIRE_THAT(compressed[5].real(), Catch::Matchers::WithinAbs(-0.250, tolerance));
    REQUIRE_THAT(compressed[5].imag(), Catch::Matchers::WithinAbs(-0.333, tolerance));

    REQUIRE_THAT(compressed[6].real(), Catch::Matchers::WithinAbs(-0.500, tolerance));
    REQUIRE_THAT(compressed[6].imag(), Catch::Matchers::WithinAbs(-0.666, tolerance));

    REQUIRE_THAT(compressed[7].real(), Catch::Matchers::WithinAbs(-0.750, tolerance));
    REQUIRE_THAT(compressed[7].imag(), Catch::Matchers::WithinAbs(-1.000, tolerance));

    auto output = stdex::mdarray<FloatComplex, stdex::dextents<size_t, 1>>{8};
    neo::fill(output.to_mdspan(), FloatComplex(1.0F, 2.0F));
    neo::add(compressed, output.to_mdspan(), output.to_mdspan());

    REQUIRE_THAT(output(0).real(), Catch::Matchers::WithinAbs(1.000, tolerance));
    REQUIRE_THAT(output(0).imag(), Catch::Matchers::WithinAbs(2.125, tolerance));

    REQUIRE_THAT(output(1).real(), Catch::Matchers::WithinAbs(1.250, tolerance));
    REQUIRE_THAT(output(1).imag(), Catch::Matchers::WithinAbs(2.333, tolerance));

    using Overlap = neo::fft::overlap_save<FloatComplex>;
    using Filter  = neo::fft::dense_filter<FloatComplex>;

    auto filter    = stdex::mdarray<FloatComplex, stdex::dextents<size_t, 2>>{2, 513};
    auto convolver = neo::fft::uniform_partitioned_convolver<Overlap, Fdl, Filter>{};
    convolver.filter(filter.to_mdspan());
}
