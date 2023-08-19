#include "compressed_float_accessor.hpp"

#include <neo/math/decibel.hpp>

#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/container: compressed_float_accessor", "", std::int8_t, std::int16_t)
{
    using Float = float;
    using stdex::mdspan;

    using raw_t  = TestType;
    using real_t = Float;

    using extents = stdex::extents<size_t, 4, 4>;
    using layout  = stdex::layout_right;

    using raw_accessor = stdex::default_accessor<raw_t>;
    using raw_mdspan   = mdspan<raw_t, extents, layout, raw_accessor>;

    using real_accessor = neo::compressed_float_accessor<real_t, raw_accessor>;
    using real_mdspan   = mdspan<real_t, extents, layout, real_accessor>;

    auto pre_scale = real_t(4);
    auto scale     = std::numeric_limits<raw_t>::max();

    auto val = real_t(1) / std::sqrt(real_t(2)) / pre_scale;
    // val = real_t(std::numbers::pi) / pre_scale;

    auto buf  = std::array<raw_t, 16>{};
    auto raw  = raw_mdspan{buf.data()};
    raw(0, 0) = static_cast<raw_t>(std::lround(val * scale));
    // raw(0,0) = static_cast<raw_t>(val * scale);

    auto real = real_mdspan{
        raw.data_handle(),
        raw.mapping(),
        real_accessor(raw.accessor()),
    };

    auto const error = val - (real(0, 0));
    std::printf("sizeof(raw_mdspan): %zu\n", sizeof(raw_mdspan));
    std::printf("sizeof(real_mdspan): %zu\n", sizeof(real_mdspan));
    std::printf("val: %.12f\n", val * pre_scale);
    std::printf("real(0, 0): %.12f\n", real(0, 0) * pre_scale);
    std::printf("error lin: %.12f\n", error);
    std::printf("error dB: %.12f\n\n", neo::to_decibels(std::abs(error)));
}
