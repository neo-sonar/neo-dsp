#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/multiply_add.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/container/sparse_matrix.hpp>
#include <neo/fft/convolution/overlap_add.hpp>
#include <neo/fft/convolution/overlap_save.hpp>
#include <neo/fft/convolution/uniform_partitioned_convolver.hpp>

namespace neo::fft {

namespace detail {
template<typename Complex>
auto dense_filter_kernel(in_vector auto x_vec, in_vector auto y_vec, inout_vector auto out_vec)
{
    auto const vec_size  = static_cast<std::ptrdiff_t>(Complex::size);
    auto const remainder = static_cast<std::ptrdiff_t>(x_vec.extent(0)) % vec_size;

    for (auto i{0}; i < remainder; ++i) {
        out_vec[i] += x_vec[i] * y_vec[i];
    }

    for (auto i{remainder}; i < static_cast<std::ptrdiff_t>(x_vec.extent(0)); i += vec_size) {
        auto x      = Complex::load_unaligned(std::next(x_vec.data_handle(), i));
        auto y      = Complex::load_unaligned(std::next(y_vec.data_handle(), i));
        auto z      = Complex::load_unaligned(std::next(out_vec.data_handle(), i));
        auto result = x * y + z;
        result.store_unaligned(std::next(out_vec.data_handle(), i));
    }
}
}  // namespace detail

template<typename Complex>
struct dense_filter
{
    using value_type = Complex;

    dense_filter() = default;

    auto filter(in_matrix auto filter) -> void { _filter = filter; }

    auto operator()(in_vector auto fdl, std::integral auto filter_index, inout_vector auto accumulator) -> void
    {
        auto const subfilter = stdex::submdspan(_filter, filter_index, stdex::full_extent);

        if constexpr (std::same_as<typename decltype(fdl)::value_type, std::complex<float>>) {

#if defined(NEO_HAS_SIMD_AVX)
            if (fdl.extent(0) - 1 > icomplex64x4::size) {
                detail::dense_filter_kernel<icomplex64x4>(fdl, subfilter, accumulator);
                return;
            }
#elif defined(NEO_HAS_SIMD_SSE2)
            if (fdl.extent(0) - 1 > icomplex64x2::size) {
                detail::dense_filter_kernel<icomplex64x2>(fdl, subfilter, accumulator);
                return;
            }
#endif
        }
        multiply_add(fdl, subfilter, accumulator, accumulator);
    }

private:
    stdex::mdspan<Complex const, stdex::dextents<size_t, 2>> _filter;
};

template<typename Float, typename Complex = std::complex<Float>>
using upols_convolver = uniform_partitioned_convolver<overlap_save<Float>, dense_fdl<Complex>, dense_filter<Complex>>;

template<typename Float, typename Complex = std::complex<Float>>
using upola_convolver = uniform_partitioned_convolver<overlap_add<Float>, dense_fdl<Complex>, dense_filter<Complex>>;

}  // namespace neo::fft
