#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/complex.hpp>
#include <neo/container/compressed_accessor.hpp>
#include <neo/container/mdspan.hpp>

#include <cmath>
#include <limits>

namespace neo::fft {

template<complex FloatComplex, complex IntComplex>
struct compressed_fdl
{
    using value_type      = FloatComplex;
    using compressed_type = IntComplex;

    compressed_fdl() = default;

    explicit compressed_fdl(stdex::dextents<size_t, 2> extents) : _fdl{extents} {}

    auto operator()(in_vector auto input, std::integral auto index) -> void
    {
        using int_type         = typename IntComplex::value_type;
        constexpr auto max_val = std::numeric_limits<int_type>::max();

        auto const fdl = stdex::submdspan(_fdl.to_mdspan(), index, stdex::full_extent);
        for (auto i{0U}; i < input.extent(0); ++i) {
            fdl[i] = IntComplex{
                static_cast<int_type>(std::lround(input[i].real() * static_cast<float>(max_val))),
                static_cast<int_type>(std::lround(input[i].imag() * static_cast<float>(max_val))),
            };
        }
    }

    auto operator()(std::integral auto index) const
    {
        auto fdl = stdex::submdspan(_fdl.to_mdspan(), index, stdex::full_extent);

        using raw_accessor        = typename decltype(fdl)::accessor_type;
        using extents             = typename decltype(fdl)::extents_type;
        using compressed_accessor = neo::compressed_accessor<FloatComplex, raw_accessor>;

        return stdex::mdspan<FloatComplex, extents, stdex::layout_right, compressed_accessor>{
            fdl.data_handle(),
            fdl.mapping(),
            compressed_accessor{fdl.accessor()},
        };
    }

private:
    stdex::mdarray<IntComplex, stdex::dextents<size_t, 2>> _fdl{};
};

}  // namespace neo::fft
