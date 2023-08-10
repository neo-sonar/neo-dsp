#pragma once

#include <neo/fft/container/mdspan.hpp>
#include <neo/fft/container/sparse_matrix.hpp>
#include <neo/fft/transform.hpp>

#include <algorithm>
#include <complex>
#include <functional>
#include <memory>
#include <vector>

namespace neo::fft {

struct sparse_upols_convolver
{
    sparse_upols_convolver() = default;

    auto filter(
        KokkosEx::mdspan<std::complex<float> const, Kokkos::dextents<size_t, 2>> filter,
        std::function<bool(std::size_t, std::size_t, std::complex<float>)> const& sparsiyFilter
    ) -> void;
    auto operator()(std::span<float> block) -> void;

private:
    std::vector<float> _window;
    std::vector<std::complex<float>> _accumulator;
    KokkosEx::mdarray<std::complex<float>, Kokkos::dextents<size_t, 2>> _fdl;
    sparse_matrix<std::complex<float>> _filter;
    std::size_t _fdlIndex{0};

    std::unique_ptr<rfft_radix2_plan<float>> _rfft;
    std::vector<std::complex<float>> _rfftBuf;
    std::vector<float> _irfftBuf;
};

template<typename T, typename U, typename V>
[[nodiscard]] constexpr auto frequency_for_bin(U windowSize, V index, double sampleRate) -> T
{
    static_assert(std::is_integral_v<U>);
    static_assert(std::is_integral_v<V>);

    return static_cast<T>(index) * static_cast<T>(sampleRate) / static_cast<T>(windowSize);
}

}  // namespace neo::fft
