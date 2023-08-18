#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/convolution/fdl_index.hpp>

namespace neo::fft {

template<typename Filter, typename Overlap>
struct uniform_partitioned_convolver
{
    using filter_type  = Filter;
    using value_type   = typename Filter::value_type;
    using overlap_type = Overlap;

    uniform_partitioned_convolver() = default;

    auto filter(in_matrix auto filter, auto... args) -> void;
    auto operator()(in_vector auto block) -> void;

private:
    Overlap _overlap{1, 1};
    Filter _filter;
    fdl_index<size_t> _indexer;
    stdex::mdarray<std::complex<value_type>, stdex::dextents<size_t, 2>> _fdl;
    stdex::mdarray<std::complex<value_type>, stdex::dextents<size_t, 1>> _accumulator;
};

template<typename Filter, typename Overlap>
auto uniform_partitioned_convolver<Filter, Overlap>::filter(in_matrix auto filter, auto... args) -> void
{
    _overlap     = Overlap{filter.extent(1) - 1, filter.extent(1) - 1};
    _indexer     = fdl_index<size_t>{filter.extent(0)};
    _fdl         = stdex::mdarray<std::complex<value_type>, stdex::dextents<size_t, 2>>{filter.extents()};
    _accumulator = stdex::mdarray<std::complex<value_type>, stdex::dextents<size_t, 1>>{filter.extent(1)};
    _filter.filter(filter, args...);
}

template<typename Filter, typename Overlap>
auto uniform_partitioned_convolver<Filter, Overlap>::operator()(in_vector auto block) -> void
{
    _overlap(block, [this](inout_vector auto inout) {
        auto copy_to_fdl = [this, inout](auto dest_idx) {
            copy(inout, stdex::submdspan(_fdl.to_mdspan(), dest_idx, stdex::full_extent));
        };

        auto multiply_delay_lines = [this](auto fdl_idx, auto filter_idx) {
            auto const fdl = stdex::submdspan(_fdl.to_mdspan(), fdl_idx, stdex::full_extent);
            _filter(fdl, filter_idx, _accumulator.to_mdspan());
        };

        fill(_accumulator.to_mdspan(), std::complex<value_type>{});
        _indexer(copy_to_fdl, multiply_delay_lines);
        copy(_accumulator.to_mdspan(), inout);
    });
}

}  // namespace neo::fft
