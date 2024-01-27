// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/add.hpp>
#include <neo/algorithm/copy.hpp>
#include <neo/algorithm/fill.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/rfft.hpp>

#include <cassert>

namespace neo::convolution {

template<complex Complex, typename Fdl, typename Filter>
struct overlap_add_convolver
{
    using value_type       = Complex;
    using real_type        = value_type_t<Complex>;
    using size_type        = std::size_t;
    using fdl_type         = Fdl;
    using filter_type      = Filter;
    using accumulator_type = typename Filter::accumulator_type;

    overlap_add_convolver() = default;

    auto filter(in_matrix_of<Complex> auto f) -> void;
    auto operator()(inout_vector_of<real_type> auto block) -> void;

private:
    size_type _block_size{2};
    size_type _num_segments{0};
    size_type _input_pos{0};
    size_type _segment_idx{0};

    fft::rfft_plan<real_type> _rfft{fft::from_order, fft::next_order(4)};

    stdex::mdarray<real_type, stdex::dextents<size_t, 1>> _window{_rfft.size()};
    stdex::mdarray<real_type, stdex::dextents<size_t, 1>> _overlap{_rfft.size()};
    stdex::mdarray<Complex, stdex::dextents<size_t, 1>> _complex_window{_rfft.size() / 2 + 1};

    Fdl _fdl;
    Filter _filter;

    accumulator_type _accumulator;
    accumulator_type _tmp_accumulator;
};

template<complex Complex, typename Fdl, typename Filter>
auto overlap_add_convolver<Complex, Fdl, Filter>::filter(in_matrix_of<Complex> auto f) -> void
{
    _block_size   = f.extent(1) - 1;
    _num_segments = f.extent(0);

    _rfft           = fft::rfft_plan<real_type>{fft::from_order, fft::next_order(_block_size * 2U)};
    _window         = stdex::mdarray<real_type, stdex::dextents<size_t, 1>>{_rfft.size()};
    _overlap        = stdex::mdarray<real_type, stdex::dextents<size_t, 1>>{_rfft.size()};
    _complex_window = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{_rfft.size() / 2 + 1};

    _fdl             = Fdl{f.extents()};
    _accumulator     = accumulator_type{f.extent(1)};
    _tmp_accumulator = accumulator_type{f.extent(1)};
    _filter.filter(f);
}

template<complex Complex, typename Fdl, typename Filter>
auto overlap_add_convolver<Complex, Fdl, Filter>::operator()(inout_vector_of<real_type> auto block) -> void
{
    assert(std::cmp_less_equal(_block_size, block.extent(0)));

    auto const num_samples = block.extent(0);
    auto num_processed     = size_type(0);

    auto window          = _window.to_mdspan();
    auto complex_window  = _complex_window.to_mdspan();
    auto overlap         = _overlap.to_mdspan();
    auto accumulator     = _accumulator.to_mdspan();
    auto tmp_accumulator = _tmp_accumulator.to_mdspan();

    while (num_processed < num_samples) {
        auto const input_was_empty = (_input_pos == 0);
        auto const num_to_process  = std::min(num_samples - num_processed, _block_size - _input_pos);

        auto const sub_block  = stdex::submdspan(block, std::tuple{num_processed, num_to_process});
        auto const sub_window = stdex::submdspan(window, std::tuple{_input_pos, num_to_process});
        copy(sub_block, sub_window);
        rfft(_rfft, window, complex_window);

        _fdl.insert(complex_window, _segment_idx);

        if (input_was_empty) {
            fill(tmp_accumulator, value_type_t<accumulator_type>{});

            auto input_index = _segment_idx;
            for (size_type filter_index{1}; filter_index < _num_segments; ++filter_index) {
                input_index += 1;
                if (input_index >= _num_segments) {
                    input_index -= _num_segments;
                }

                _filter(_fdl[input_index], filter_index, tmp_accumulator);
            }
        }

        copy(tmp_accumulator, accumulator);
        _filter(_fdl[_segment_idx], 0, accumulator);

        irfft(_rfft, complex_window, window);
        scale(real_type(1) / static_cast<real_type>(_rfft.size()), window);

        // Add overlap
        auto sub_overlap = stdex::submdspan(overlap, std::tuple{_input_pos, num_to_process});
        add(sub_window, sub_overlap, sub_block);

        // Input buffer full => Next block
        _input_pos += num_to_process;

        if (_input_pos == _block_size) {
            fill(window, real_type{});
            _input_pos = 0;

            auto window_save  = stdex::submdspan(window, std::tuple{_block_size, _block_size + _block_size});
            auto overlap_save = stdex::submdspan(overlap, std::tuple{0, _block_size});
            copy(window_save, overlap_save);

            _segment_idx = (_segment_idx > 0) ? (_segment_idx - 1) : (_num_segments - 1);
        }

        num_processed += num_to_process;
    }
}

}  // namespace neo::convolution
