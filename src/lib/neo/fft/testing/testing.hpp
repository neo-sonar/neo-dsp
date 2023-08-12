#pragma once

#include <neo/fft/config.hpp>

#include <neo/fft/container/mdspan.hpp>
#include <neo/fft/math/complex.hpp>

#include <algorithm>
#include <concepts>
#include <cstdio>
#include <filesystem>
#include <random>
#include <vector>

namespace neo::fft {

template<float_or_complex FloatOrComplex, typename URNG = std::mt19937>
[[nodiscard]] auto generate_noise_signal(std::size_t length, typename URNG::result_type seed)
{
    using Float = real_or_complex_value_t<FloatOrComplex>;

    auto rng    = URNG{seed};
    auto dist   = std::uniform_real_distribution<Float>{Float(-1), Float(1)};
    auto signal = std::vector<FloatOrComplex>(length);

    if constexpr (std::floating_point<FloatOrComplex>) {
        std::generate(signal.begin(), signal.end(), [&] { return dist(rng); });
    } else {
        std::generate(signal.begin(), signal.end(), [&] { return FloatOrComplex{dist(rng), dist(rng)}; });
    }

    return signal;
}

template<std::floating_point Float>
[[nodiscard]] auto generate_identity_impulse(std::size_t blockSize, std::size_t numPartitions)
    -> KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<std::size_t, 2>>
{
    auto const windowSize = blockSize * 2;
    auto const numBins    = windowSize / 2 + 1;

    auto impulse = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<std::size_t, 2>>{
        numPartitions,
        numBins,
    };

    for (auto partition{0U}; partition < impulse.extent(0); ++partition) {
        for (auto bin{0U}; bin < impulse.extent(1); ++bin) {
            impulse(partition, bin) = std::complex{Float(1), Float(0)};
        }
    }
    return impulse;
}

}  // namespace neo::fft
