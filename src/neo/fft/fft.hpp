#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/backend/fallback.hpp>
#include <neo/fft/bitrevorder.hpp>

#if defined(NEO_PLATFORM_APPLE)
    #include <neo/fft/backend/accelerate.hpp>
#endif

#if defined(NEO_HAS_INTEL_IPP)
    #include <neo/fft/backend/ipp.hpp>
#endif

namespace neo::fft {

template<typename Plan, inout_vector Vec>
constexpr auto fft(Plan& plan, Vec inout) -> void
{
    plan(inout, direction::forward);
}

template<typename Plan, in_vector InVec, out_vector OutVec>
constexpr auto fft(Plan& plan, InVec input, OutVec output) -> void
{
    copy(input, output);
    fft(plan, output);
}

template<typename Plan, inout_vector Vec>
constexpr auto ifft(Plan& plan, Vec inout) -> void
{
    plan(inout, direction::backward);
}

template<typename Plan, in_vector InVec, out_vector OutVec>
constexpr auto ifft(Plan& plan, InVec input, OutVec output) -> void
{
    copy(input, output);
    ifft(plan, output);
}

#if defined(NEO_PLATFORM_APPLE)
template<complex Complex>
using fft_plan = apple_vdsp_fft_plan<Complex>;
#elif defined(NEO_HAS_INTEL_IPP)
template<complex Complex>
using fft_plan = intel_ipp_fft_plan<Complex>;
#else
template<complex Complex>
using fft_plan = fallback_fft_plan<Complex>;
#endif

}  // namespace neo::fft
