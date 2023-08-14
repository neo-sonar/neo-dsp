#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/algorithm/fill.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/transform.hpp>
#include <neo/math/complex.hpp>
#include <neo/math/divide_round_up.hpp>
#include <neo/math/ilog2.hpp>
#include <neo/math/next_power_of_two.hpp>

#include <functional>

namespace neo::fft {

template<std::floating_point Float>
struct overlap_add
{
    using real_type    = Float;
    using complex_type = std::complex<Float>;
    using size_type    = std::size_t;

    overlap_add(size_type block_size, size_type filter_size);

    [[nodiscard]] auto block_size() const noexcept -> size_type;
    [[nodiscard]] auto filter_size() const noexcept -> size_type;
    [[nodiscard]] auto transform_size() const noexcept -> size_type;
    [[nodiscard]] auto overlaps() const noexcept -> size_type;

    auto operator()(inout_vector auto block, auto callback) -> void;

private:
    size_type _block_size;
    size_type _filter_size;

    rfft_radix2_plan<Float> _rfft{ilog2(next_power_of_two(_block_size + _filter_size - 1UL))};
    stdex::mdarray<Float, stdex::dextents<size_t, 1>> _real_buffer{_rfft.size()};
    stdex::mdarray<complex_type, stdex::dextents<size_t, 1>> _complex_buffer{_rfft.size()};

    size_type _overlapIdx{0};
    stdex::mdarray<Float, stdex::dextents<size_t, 2>> _overlaps;
};

template<std::floating_point Float>
overlap_add<Float>::overlap_add(size_type block_size, size_type filter_size)
    : _block_size{block_size}
    , _filter_size{filter_size}
    , _overlaps{
          divide_round_up(_block_size + _filter_size - 1UL, _block_size),
          _block_size + _filter_size - 1UL,
      }
{}

template<std::floating_point Float>
auto overlap_add<Float>::block_size() const noexcept -> size_type
{
    return _block_size;
}

template<std::floating_point Float>
auto overlap_add<Float>::filter_size() const noexcept -> size_type
{
    return _filter_size;
}

template<std::floating_point Float>
auto overlap_add<Float>::transform_size() const noexcept -> size_type
{
    return _rfft.size();
}

template<std::floating_point Float>
auto overlap_add<Float>::overlaps() const noexcept -> size_type
{
    return _overlaps.extent(0);
}

template<std::floating_point Float>
auto overlap_add<Float>::operator()(inout_vector auto block, auto callback) -> void
{
    NEO_EXPECTS(_rfft.size() == block.extent(0) * 2U);

    auto const real_buffer    = _real_buffer.to_mdspan();
    auto const complex_buffer = _complex_buffer.to_mdspan();

    fill(real_buffer, Float(0));
    copy(block, stdex::submdspan(real_buffer, std::tuple{0, block.extent(1)}));

    _rfft(real_buffer, complex_buffer);

    auto const alpha  = 1.0F / static_cast<Float>(_rfft.size());
    auto const coeffs = stdex::submdspan(complex_buffer, std::tuple{0, _rfft.size() / 2 + 1});
    scale(alpha, coeffs);

    callback(coeffs);

    _rfft(complex_buffer, real_buffer);

    copy(
        stdex::submdspan(real_buffer, std::tuple{0, _overlaps.extent(1)}),
        stdex::submdspan(_overlaps.to_mdspan(), _overlapIdx, stdex::full_extent)
    );
    copy(stdex::submdspan(real_buffer, std::tuple{0, block.extent(0)}), block);

    ++_overlapIdx;
    if (_overlapIdx == _overlaps.extent(0)) {
        _overlapIdx = 0;
    }
}

}  // namespace neo::fft
