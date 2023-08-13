#pragma once

#include <neo/container/mdspan.hpp>
#include <neo/math/complex.hpp>

#include <concepts>

namespace neo::fft {

template<in_vector InVec>
    requires complex<typename InVec::value_type>
struct conjugate_view
{
    using value_type = typename InVec::value_type;

    constexpr explicit conjugate_view(InVec twiddles) noexcept : _twiddles{twiddles} {}

    [[nodiscard]] constexpr auto operator[](std::integral auto index) const -> value_type
    {
        using std::conj;
        return conj(_twiddles[index]);
    }

private:
    InVec _twiddles;
};

}  // namespace neo::fft
