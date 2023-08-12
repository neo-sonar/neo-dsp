#pragma once

#include <neo/fft/config.hpp>

#include <concepts>

namespace neo::fft {

template<std::integral Integral>
[[nodiscard]] constexpr auto ilog2(Integral x) -> Integral
{
    NEO_FFT_PRECONDITION(x > Integral(0));

    auto result = Integral{0};
    for (; x > Integral(1); x >>= Integral(1)) {
        ++result;
    }
    return result;
}

}  // namespace neo::fft
