// SPDX-License-Identifier: MIT

#include "compressed_accessor.hpp"

#include <neo/algorithm/add.hpp>
#include <neo/complex/scalar_complex.hpp>
#include <neo/unit/decibel.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEMPLATE_TEST_CASE("neo/container: compressed_accessor", "", std::int8_t, std::int16_t)
{
    using Int   = TestType;
    using Float = float;

    using extents = stdex::extents<size_t, 4, 4>;
    using layout  = stdex::layout_right;

    using raw_accessor = stdex::default_accessor<Int>;
    using raw_mdspan   = stdex::mdspan<Int, extents, layout, raw_accessor>;

    using real_accessor = neo::compressed_accessor<Float, raw_accessor>;
    using real_mdspan   = stdex::mdspan<Float, extents, layout, real_accessor>;

    static constexpr auto scale     = std::numeric_limits<Int>::max();
    static constexpr auto tolerance = [] {
        if constexpr (sizeof(Int) == 1) {
            return 0.1;
        }
        return 0.001;
    }();

    auto buf  = std::array<Int, 16>{};
    auto raw  = raw_mdspan{buf.data()};
    auto real = real_mdspan{raw.data_handle(), raw.mapping(), real_accessor(raw.accessor())};

    raw(0, 0) = static_cast<Int>(std::lround(Float(0.0) * scale));
    raw(0, 1) = static_cast<Int>(std::lround(Float(0.1) * scale));
    raw(0, 2) = static_cast<Int>(std::lround(Float(0.2) * scale));
    raw(0, 3) = static_cast<Int>(std::lround(Float(0.3) * scale));

    REQUIRE_THAT(real(0, 0), Catch::Matchers::WithinAbs(0.0, tolerance));
    REQUIRE_THAT(real(0, 1), Catch::Matchers::WithinAbs(0.1, tolerance));
    REQUIRE_THAT(real(0, 2), Catch::Matchers::WithinAbs(0.2, tolerance));
    REQUIRE_THAT(real(0, 3), Catch::Matchers::WithinAbs(0.3, tolerance));

    auto other  = stdex::mdarray<Float, extents, layout>{};
    other(0, 0) = Float(0.5);
    other(0, 1) = Float(0.5);
    other(0, 2) = Float(0.5);
    other(0, 3) = Float(0.5);

    neo::add(real, other.to_mdspan(), other.to_mdspan());
    REQUIRE_THAT(other(0, 0), Catch::Matchers::WithinAbs(0.0 + 0.5, tolerance));
    REQUIRE_THAT(other(0, 1), Catch::Matchers::WithinAbs(0.1 + 0.5, tolerance));
    REQUIRE_THAT(other(0, 2), Catch::Matchers::WithinAbs(0.2 + 0.5, tolerance));
    REQUIRE_THAT(other(0, 3), Catch::Matchers::WithinAbs(0.3 + 0.5, tolerance));
}

TEMPLATE_PRODUCT_TEST_CASE("neo/container: compressed_accessor", "", (neo::scalar_complex), (std::int8_t, std::int16_t))
{
    using CompressedComplex = TestType;
    using Int               = typename CompressedComplex::value_type;
    using FloatComplex      = std::complex<float>;
    using Float             = float;

    using extents = stdex::extents<size_t, 4, 4>;
    using layout  = stdex::layout_right;

    using raw_accessor = stdex::default_accessor<CompressedComplex>;
    using raw_mdspan   = stdex::mdspan<CompressedComplex, extents, layout, raw_accessor>;

    using real_accessor = neo::compressed_accessor<FloatComplex, raw_accessor>;
    using real_mdspan   = stdex::mdspan<FloatComplex, extents, layout, real_accessor>;

    static constexpr auto scale     = std::numeric_limits<Int>::max();
    static constexpr auto tolerance = [] {
        if constexpr (sizeof(Int) == 1) {
            return 0.1;
        }
        return 0.001;
    }();

    auto buf  = std::array<CompressedComplex, 16>{};
    auto raw  = raw_mdspan{buf.data()};
    auto real = real_mdspan{raw.data_handle(), raw.mapping(), real_accessor(raw.accessor())};

    raw(0, 0) = CompressedComplex{static_cast<Int>(std::lround(Float(0.0) * scale))};
    raw(0, 1) = CompressedComplex{static_cast<Int>(std::lround(Float(0.1) * scale))};
    raw(0, 2) = CompressedComplex{static_cast<Int>(std::lround(Float(0.2) * scale))};
    raw(0, 3) = CompressedComplex{static_cast<Int>(std::lround(Float(0.3) * scale))};

    REQUIRE_THAT(real(0, 0).real(), Catch::Matchers::WithinAbs(0.0, tolerance));
    REQUIRE_THAT(real(0, 1).real(), Catch::Matchers::WithinAbs(0.1, tolerance));
    REQUIRE_THAT(real(0, 2).real(), Catch::Matchers::WithinAbs(0.2, tolerance));
    REQUIRE_THAT(real(0, 3).real(), Catch::Matchers::WithinAbs(0.3, tolerance));

    auto other  = stdex::mdarray<FloatComplex, extents, layout>{};
    other(0, 0) = FloatComplex(0.5);
    other(0, 1) = FloatComplex(0.5);
    other(0, 2) = FloatComplex(0.5);
    other(0, 3) = FloatComplex(0.5);

    neo::add(real, other.to_mdspan(), other.to_mdspan());
    REQUIRE_THAT(other(0, 0).real(), Catch::Matchers::WithinAbs(0.0 + 0.5, tolerance));
    REQUIRE_THAT(other(0, 1).real(), Catch::Matchers::WithinAbs(0.1 + 0.5, tolerance));
    REQUIRE_THAT(other(0, 2).real(), Catch::Matchers::WithinAbs(0.2 + 0.5, tolerance));
    REQUIRE_THAT(other(0, 3).real(), Catch::Matchers::WithinAbs(0.3 + 0.5, tolerance));
}
