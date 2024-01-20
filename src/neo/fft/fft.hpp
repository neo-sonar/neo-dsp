// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/order.hpp>

#include <neo/fft/reference/c2c_dif3_plan.hpp>
#include <neo/fft/reference/c2c_dif5_plan.hpp>
#include <neo/fft/reference/c2c_dit2_plan.hpp>
#include <neo/fft/reference/c2c_dit4_plan.hpp>
#include <neo/fft/reference/c2c_stockham_dif2_plan.hpp>
#include <neo/fft/reference/c2c_stockham_dif3_plan.hpp>
#include <neo/fft/reference/c2c_stockham_dif4_plan.hpp>
#include <neo/fft/reference/c2c_stockham_dif5_plan.hpp>
#include <neo/fft/reference/c2c_stockham_dif8_plan.hpp>
#include <neo/fft/reference/c2c_stockham_dit4_plan.hpp>

#if defined(NEO_HAS_APPLE_ACCELERATE)
    #include <neo/fft/backend/vdsp.hpp>
#endif

#if defined(NEO_HAS_INTEL_IPP)
    #include <neo/fft/backend/ipp.hpp>
#endif

#if defined(NEO_HAS_INTEL_MKL)
    #include <neo/fft/backend/mkl.hpp>
#endif

namespace neo::fft {

#if defined(NEO_HAS_APPLE_ACCELERATE)
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
using fft_plan = c2c_dit2_plan<Complex>;
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

}  // namespace neo::fft
