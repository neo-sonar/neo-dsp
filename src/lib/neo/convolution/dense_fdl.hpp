#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>

namespace neo {

template<typename Complex>
struct dense_fdl
{
    using value_type       = Complex;
    using accumulator_type = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>;

    dense_fdl() = default;

    explicit dense_fdl(stdex::dextents<size_t, 2> extents) noexcept : _fdl{extents} {}

    auto operator()(in_vector auto input, std::integral auto index) noexcept -> void
    {
        copy(input, stdex::submdspan(_fdl.to_mdspan(), index, stdex::full_extent));
    }

    [[nodiscard]] auto operator()(std::integral auto index) const noexcept -> in_vector auto
    {
        return stdex::submdspan(_fdl.to_mdspan(), index, stdex::full_extent);
    }

private:
    stdex::mdarray<Complex, stdex::dextents<size_t, 2>> _fdl{};
};

}  // namespace neo
