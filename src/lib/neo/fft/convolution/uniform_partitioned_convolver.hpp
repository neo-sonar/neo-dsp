#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/convolution/frequency_delay_line.hpp>

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
    frequency_delay_line<std::complex<value_type>> _fdl;
    stdex::mdarray<std::complex<value_type>, stdex::dextents<size_t, 1>> _accumulator;
};

template<typename Filter, typename Overlap>
auto uniform_partitioned_convolver<Filter, Overlap>::filter(in_matrix auto filter, auto... args) -> void
{
    _overlap     = Overlap{filter.extent(1) - 1, filter.extent(1) - 1};
    _fdl         = frequency_delay_line<std::complex<value_type>>{filter.extents()};
    _accumulator = stdex::mdarray<std::complex<value_type>, stdex::dextents<size_t, 1>>{filter.extent(1)};
    _filter.filter(filter, args...);
}

template<typename Filter, typename Overlap>
auto uniform_partitioned_convolver<Filter, Overlap>::operator()(in_vector auto block) -> void
{
    _overlap(block, [this](inout_vector auto inout) {
        fill(_accumulator.to_mdspan(), std::complex<value_type>{});

        _fdl(inout, [this](in_vector auto delay_line, std::integral auto filter_row) {
            _filter(delay_line, filter_row, _accumulator.to_mdspan());
        });

        copy(_accumulator.to_mdspan(), inout);
    });
}

}  // namespace neo::fft
