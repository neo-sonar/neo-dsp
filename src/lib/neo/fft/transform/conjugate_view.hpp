#pragma once

#include <neo/fft/math/complex.hpp>

#include <span>

namespace neo::fft {

template<typename Complex, std::size_t Extent = std::dynamic_extent>
struct conjugate_view
{
    constexpr explicit conjugate_view(std::span<Complex const, Extent> twiddles) noexcept : _twiddles{twiddles} {}

    [[nodiscard]] constexpr auto operator[](std::size_t idx) const noexcept -> Complex
    {
        return std::conj(_twiddles[idx]);
    }

private:
    std::span<Complex const, Extent> _twiddles;
};

}  // namespace neo::fft
