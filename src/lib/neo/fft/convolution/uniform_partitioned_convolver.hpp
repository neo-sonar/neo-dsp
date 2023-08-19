#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/convolution/fdl_index.hpp>

namespace neo::fft {

template<typename Complex>
struct dense_fdl
{
    using value_type = Complex;

    dense_fdl() = default;

    explicit dense_fdl(stdex::dextents<size_t, 2> extents) : _fdl{extents} {}

    auto operator()(in_vector auto input, std::integral auto index) -> void
    {
        copy(input, stdex::submdspan(_fdl.to_mdspan(), index, stdex::full_extent));
    }

    auto operator()(std::integral auto index) const
    {
        return stdex::submdspan(_fdl.to_mdspan(), index, stdex::full_extent);
    }

private:
    stdex::mdarray<Complex, stdex::dextents<size_t, 2>> _fdl{};
};

template<typename Overlap, typename Fdl, typename Filter>
struct uniform_partitioned_convolver
{
    using overlap_type = Overlap;
    using fdl_type     = Fdl;
    using filter_type  = Filter;
    using value_type   = typename Filter::value_type;

    uniform_partitioned_convolver() = default;

    auto filter(in_matrix auto filter, auto... args) -> void;
    auto operator()(in_vector auto block) -> void;

private:
    Overlap _overlap{1, 1};

    Fdl _fdl;
    fdl_index<size_t> _indexer;

    Filter _filter;
    stdex::mdarray<std::complex<value_type>, stdex::dextents<size_t, 1>> _accumulator;
};

template<typename Overlap, typename Fdl, typename Filter>
auto uniform_partitioned_convolver<Overlap, Fdl, Filter>::filter(in_matrix auto filter, auto... args) -> void
{
    _overlap     = Overlap{filter.extent(1) - 1, filter.extent(1) - 1};
    _indexer     = fdl_index<size_t>{filter.extent(0)};
    _fdl         = Fdl{filter.extents()};
    _accumulator = stdex::mdarray<std::complex<value_type>, stdex::dextents<size_t, 1>>{filter.extent(1)};
    _filter.filter(filter, args...);
}

template<typename Overlap, typename Fdl, typename Filter>
auto uniform_partitioned_convolver<Overlap, Fdl, Filter>::operator()(in_vector auto block) -> void
{
    _overlap(block, [this](inout_vector auto inout) {
        fill(_accumulator.to_mdspan(), std::complex<value_type>{});

        auto insert   = [this, inout](auto index) { _fdl(inout, index); };
        auto multiply = [this](auto fdl, auto filter) { _filter(_fdl(fdl), filter, _accumulator.to_mdspan()); };
        _indexer(insert, multiply);

        copy(_accumulator.to_mdspan(), inout);
    });
}

}  // namespace neo::fft
