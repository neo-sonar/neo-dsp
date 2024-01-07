// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex.hpp>
#include <neo/fft/fallback/fallback_rfft_plan.hpp>
#include <neo/fft/fft.hpp>
#include <neo/math/conj.hpp>

namespace neo::fft {

#if defined(NEO_HAS_INTEL_IPP)
template<std::floating_point Float, typename Complex = std::complex<Float>>
using rfft_plan = intel_ipp_rfft_plan<Float, Complex>;
#else
template<std::floating_point Float, typename Complex = std::complex<Float>>
using rfft_plan = fallback_rfft_plan<Float, Complex>;
#endif

template<typename Plan, in_vector InVec, out_vector OutVec>
    requires(std::floating_point<value_type_t<InVec>> and complex<value_type_t<OutVec>>)
constexpr auto rfft(Plan& plan, InVec input, OutVec output)
{
    return plan(input, output);
}

template<typename Plan, in_vector InVec, out_vector OutVec>
    requires(complex<value_type_t<InVec>> and std::floating_point<value_type_t<OutVec>>)
constexpr auto irfft(Plan& plan, InVec input, OutVec output)
{
    return plan(input, output);
}

template<in_vector InVec, out_vector OutVecA, out_vector OutVecB>
auto rfft_deinterleave(InVec dft, OutVecA a, OutVecB b) -> void
{
    using Complex = typename InVec::value_type;
    using Float   = typename Complex::value_type;

    auto const n = dft.size();
    auto const i = Complex{Float(0), Float(-1)};

    a[0] = dft[0].real();
    b[0] = dft[0].imag();

    for (auto k{1U}; k < n / 2 + 1; ++k) {
        a[k] = (dft[k] + math::conj(dft[n - k])) * Float(0.5);
        b[k] = (i * (dft[k] - math::conj(dft[n - k]))) * Float(0.5);
    }
}

}  // namespace neo::fft
