// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/fallback/fallback_fft_plan.hpp>
#include <neo/math/bit_ceil.hpp>
#include <neo/math/ilog2.hpp>
#include <neo/type_traits/value_type_t.hpp>

#if defined(NEO_HAS_APPLE_VDSP)
    #include <neo/fft/backend/vdsp.hpp>
#endif

#if defined(NEO_HAS_INTEL_IPP)
    #include <neo/fft/backend/ipp.hpp>
#endif

#if defined(NEO_HAS_INTEL_MKL)
    #include <neo/fft/backend/mkl.hpp>
#endif

namespace neo::fft {

#if defined(NEO_HAS_APPLE_VDSP)
template<complex Complex>
using fft_plan = apple_vdsp_fft_plan<Complex>;
#elif defined(NEO_HAS_INTEL_IPP)
template<complex Complex>
using fft_plan = intel_ipp_fft_plan<Complex>;
#elif defined(NEO_HAS_INTEL_MKL)
template<complex Complex>
using fft_plan = intel_mkl_fft_plan<Complex>;
#else
template<complex Complex>
using fft_plan = fallback_fft_plan<Complex>;
#endif

template<typename Plan, inout_vector Vec>
constexpr auto fft(Plan& plan, Vec inout) -> void
{
    plan(inout, direction::forward);
}

template<typename Plan, in_vector InVec, out_vector OutVec>
constexpr auto fft(Plan& plan, InVec input, OutVec output) -> void
{
    if constexpr (requires { plan(input, output, direction::forward); }) {
        plan(input, output, direction::forward);
    } else {
        copy(input, output);
        fft(plan, output);
    }
}

template<typename Plan, inout_vector Vec>
    requires std::floating_point<value_type_t<Vec>>
constexpr auto fft(Plan& plan, split_complex<Vec> inout) -> void
{
    plan(inout, direction::forward);
}

template<typename Plan, in_vector InVec, out_vector OutVec>
    requires std::same_as<value_type_t<InVec>, value_type_t<OutVec>>
constexpr auto fft(Plan& plan, split_complex<InVec> in, split_complex<OutVec> out) -> void
{
    plan(in, out, direction::forward);
}

template<typename Plan, inout_vector Vec>
constexpr auto ifft(Plan& plan, Vec inout) -> void
{
    plan(inout, direction::backward);
}

template<typename Plan, in_vector InVec, out_vector OutVec>
constexpr auto ifft(Plan& plan, InVec input, OutVec output) -> void
{
    if constexpr (requires { plan(input, output, direction::backward); }) {
        plan(input, output, direction::backward);
    } else {
        copy(input, output);
        ifft(plan, output);
    }
}

template<typename Plan, inout_vector Vec>
    requires std::floating_point<value_type_t<Vec>>
constexpr auto ifft(Plan& plan, split_complex<Vec> inout) -> void
{
    plan(inout, direction::backward);
}

template<typename Plan, in_vector InVec, out_vector OutVec>
    requires std::same_as<value_type_t<InVec>, value_type_t<OutVec>>
constexpr auto ifft(Plan& plan, split_complex<InVec> in, split_complex<OutVec> out) -> void
{
    plan(in, out, direction::backward);
}

template<std::integral Int>
[[nodiscard]] constexpr auto next_order(Int size) noexcept -> Int
{
    auto const usize = static_cast<std::make_unsigned_t<Int>>(size);
    return static_cast<Int>(ilog2(bit_ceil(usize)));
}

}  // namespace neo::fft
