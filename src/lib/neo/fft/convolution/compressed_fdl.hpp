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

    explicit compressed_fdl(stdex::dextents<size_t, 2> extents) noexcept : _fdl{extents} {}

    auto operator()(in_vector auto input, std::integral auto index) noexcept -> void
    {
        auto const compress = [](auto val) {
            using int_type         = typename IntComplex::value_type;
            constexpr auto max_val = std::numeric_limits<int_type>::max();
            return static_cast<int_type>(std::lround(val * static_cast<float>(max_val)));
        };

        auto const fdl = stdex::submdspan(_fdl.to_mdspan(), index, stdex::full_extent);
        for (auto i{0U}; i < input.extent(0); ++i) {
            fdl[i] = IntComplex{compress(input[i].real()), compress(input[i].imag())};
        }
    }

    [[nodiscard]] auto operator()(std::integral auto index) const noexcept -> in_vector auto
    {
        auto const subfilter = stdex::submdspan(_fdl.to_mdspan(), index, stdex::full_extent);
        return stdex::mdspan{
            subfilter.data_handle(),
            subfilter.mapping(),
            compressed_accessor<FloatComplex, typename decltype(subfilter)::accessor_type>{subfilter.accessor()},
        };
    }

private:
    stdex::mdarray<IntComplex, stdex::dextents<size_t, 2>> _fdl{};
};

}  // namespace neo::fft
