// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/fallback/fallback_split_fft_plan.hpp>
#include <neo/type_traits/value_type_t.hpp>

#if defined(NEO_HAS_APPLE_ACCELERATE)
    #include <neo/fft/backend/vdsp.hpp>
#endif

#if defined(NEO_HAS_INTEL_IPP)
    #include <neo/fft/backend/ipp.hpp>
#endif

namespace neo::fft {

#if defined(NEO_HAS_APPLE_ACCELERATE)
/// \ingroup neo-fft
template<std::floating_point Float>
using split_fft_plan = apple_vdsp_split_fft_plan<Float>;
#elif defined(NEO_HAS_INTEL_IPP)
/// \ingroup neo-fft
template<std::floating_point Float>
using split_fft_plan = intel_ipp_split_fft_plan<Float>;
#else
/// \ingroup neo-fft
template<std::floating_point Float>
using split_fft_plan = fallback_split_fft_plan<Float>;
#endif

/// \ingroup neo-fft
template<typename Plan, inout_vector Vec>
    requires std::floating_point<value_type_t<Vec>>
constexpr auto fft(Plan& plan, split_complex<Vec> inout) -> void
{
    plan(inout, direction::forward);
}

/// \ingroup neo-fft
template<typename Plan, in_vector InVec, out_vector OutVec>
    requires std::same_as<value_type_t<InVec>, value_type_t<OutVec>>
constexpr auto fft(Plan& plan, split_complex<InVec> in, split_complex<OutVec> out) -> void
{
    plan(in, out, direction::forward);
}

/// \ingroup neo-fft
template<typename Plan, inout_vector Vec>
    requires std::floating_point<value_type_t<Vec>>
constexpr auto ifft(Plan& plan, split_complex<Vec> inout) -> void
{
    plan(inout, direction::backward);
}

/// \ingroup neo-fft
template<typename Plan, in_vector InVec, out_vector OutVec>
    requires std::same_as<value_type_t<InVec>, value_type_t<OutVec>>
constexpr auto ifft(Plan& plan, split_complex<InVec> in, split_complex<OutVec> out) -> void
{
    plan(in, out, direction::backward);
}

}  // namespace neo::fft
