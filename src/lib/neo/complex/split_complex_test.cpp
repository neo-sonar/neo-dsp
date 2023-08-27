#include "split_complex.hpp"

#include <neo/algorithm/fill.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/complex: split_complex", "", float, double)
{
    using Float = TestType;

    auto const size = std::size_t(42);

    auto buffer = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{2, size};
    auto split  = neo::split_complex{
        stdex::submdspan(buffer.to_mdspan(), 0, stdex::full_extent),
        stdex::submdspan(buffer.to_mdspan(), 1, stdex::full_extent),
    };

    REQUIRE(split.real.extent(0) == 42UL);
    REQUIRE(split.imag.extent(0) == 42UL);

    SECTION("add")
    {
        auto x_buffer   = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{2, size};
        auto y_buffer   = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{2, size};
        auto out_buffer = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{2, size};

        auto x = neo::split_complex{
            stdex::submdspan(x_buffer.to_mdspan(), 0, stdex::full_extent),
            stdex::submdspan(x_buffer.to_mdspan(), 1, stdex::full_extent),
        };
        auto y = neo::split_complex{
            stdex::submdspan(y_buffer.to_mdspan(), 0, stdex::full_extent),
            stdex::submdspan(y_buffer.to_mdspan(), 1, stdex::full_extent),
        };
        auto out = neo::split_complex{
            stdex::submdspan(out_buffer.to_mdspan(), 0, stdex::full_extent),
            stdex::submdspan(out_buffer.to_mdspan(), 1, stdex::full_extent),
        };

        neo::fill(x.real, Float(1));
        neo::fill(x.imag, Float(2));

        neo::fill(y.real, Float(3));
        neo::fill(y.imag, Float(4));

        neo::add(x, y, out);

        for (auto i{0}; i < static_cast<int>(out.real.extent(0)); ++i) {
            REQUIRE(out.real[i] == Catch::Approx(4.0));
            REQUIRE(out.imag[i] == Catch::Approx(6.0));
        }
    }

    SECTION("multiply_add")
    {
        auto x_buffer   = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{2, size};
        auto y_buffer   = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{2, size};
        auto z_buffer   = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{2, size};
        auto out_buffer = stdex::mdarray<Float, stdex::dextents<size_t, 2>>{2, size};

        auto x = neo::split_complex{
            stdex::submdspan(x_buffer.to_mdspan(), 0, stdex::full_extent),
            stdex::submdspan(x_buffer.to_mdspan(), 1, stdex::full_extent),
        };
        auto y = neo::split_complex{
            stdex::submdspan(y_buffer.to_mdspan(), 0, stdex::full_extent),
            stdex::submdspan(y_buffer.to_mdspan(), 1, stdex::full_extent),
        };
        auto z = neo::split_complex{
            stdex::submdspan(z_buffer.to_mdspan(), 0, stdex::full_extent),
            stdex::submdspan(z_buffer.to_mdspan(), 1, stdex::full_extent),
        };
        auto out = neo::split_complex{
            stdex::submdspan(out_buffer.to_mdspan(), 0, stdex::full_extent),
            stdex::submdspan(out_buffer.to_mdspan(), 1, stdex::full_extent),
        };

        neo::fill(x.real, Float(1));
        neo::fill(x.imag, Float(2));

        neo::fill(y.real, Float(3));
        neo::fill(y.imag, Float(4));

        neo::fill(z.real, Float(5));
        neo::fill(z.imag, Float(6));

        neo::multiply_add(x, y, z, out);

        for (auto i{0}; i < static_cast<int>(out.real.extent(0)); ++i) {
            REQUIRE(out.real[i] == Catch::Approx(0.0));
            REQUIRE(out.imag[i] == Catch::Approx(16.0));
        }
    }
}
