#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>

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
    stdex::mdarray<std::complex<value_type>, stdex::dextents<size_t, 1>> _accumulator;
    stdex::mdarray<std::complex<value_type>, stdex::dextents<size_t, 2>> _fdl;
    std::size_t _fdl_write_pos{0};
};

template<typename Filter, typename Overlap>
auto uniform_partitioned_convolver<Filter, Overlap>::filter(in_matrix auto filter, auto... args) -> void
{
    _fdl_write_pos = 0UL;
    _overlap       = Overlap{filter.extent(1) - 1, filter.extent(1) - 1};
    _fdl           = stdex::mdarray<std::complex<value_type>, stdex::dextents<size_t, 2>>{filter.extents()};
    _accumulator   = stdex::mdarray<std::complex<value_type>, stdex::dextents<size_t, 1>>{filter.extent(1)};
    _filter.filter(filter, args...);
}

template<typename Filter, typename Overlap>
auto uniform_partitioned_convolver<Filter, Overlap>::operator()(in_vector auto block) -> void
{
    _overlap(block, [this](inout_vector auto inout) {
        auto const fdl         = _fdl.to_mdspan();
        auto const accumulator = _accumulator.to_mdspan();

        copy(inout, stdex::submdspan(fdl, _fdl_write_pos, stdex::full_extent));
        _filter(fdl, _fdl_write_pos, accumulator);
        copy(accumulator, inout);

        ++_fdl_write_pos;
        if (_fdl_write_pos == fdl.extent(0)) {
            _fdl_write_pos = 0;
        }
    });
}

}  // namespace neo::fft
