#pragma once

#include <neo/config.hpp>

#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/container/sparse_matrix.hpp>
#include <neo/fft/convolution/multiply_add.hpp>
#include <neo/fft/convolution/overlap_add.hpp>
#include <neo/fft/convolution/overlap_save.hpp>
#include <neo/fft/convolution/uniform_partitioned_convolver.hpp>

namespace neo::fft {

template<typename Float>
struct dense_filter
{
    using value_type = Float;

    dense_filter() = default;

    auto filter(in_matrix auto filter) -> void { _filter = filter; }

    auto operator()(in_vector auto fdl, std::integral auto filter_index, inout_vector auto accumulator) -> void
    {
        auto const subfilter = stdex::submdspan(_filter, filter_index, stdex::full_extent);

#if defined(NEO_HAS_SIMD_SSE2)
        if constexpr (std::same_as<typename decltype(fdl)::value_type, std::complex<float>>) {
            if (fdl.extent(0) - 1 > icomplex64x2::size) {
                NEO_EXPECTS(((fdl.extent(0) - 1U) % icomplex64x2::size) == 0);
                accumulator[0] = fdl[0] * subfilter[0] + accumulator[0];
                for (auto i{1UL}; i < fdl.extent(0); i += icomplex64x2::size) {
                    auto x      = icomplex64x2::load_unaligned(std::next(fdl.data_handle(), i));
                    auto y      = icomplex64x2::load_unaligned(std::next(subfilter.data_handle(), i));
                    auto z      = icomplex64x2::load_unaligned(std::next(accumulator.data_handle(), i));
                    auto result = x * y + z;
                    result.store_unaligned(std::next(accumulator.data_handle(), i));
                }
                return;
            }
        }
#endif
        multiply_add(fdl, subfilter, accumulator, accumulator);
    }

private:
    stdex::mdspan<std::complex<Float> const, stdex::dextents<size_t, 2>> _filter;
};

template<typename Float>
using upols_convolver = uniform_partitioned_convolver<dense_filter<Float>, overlap_save<Float>>;

template<typename Float>
using upola_convolver = uniform_partitioned_convolver<dense_filter<Float>, overlap_add<Float>>;

}  // namespace neo::fft
