#pragma once

#include <span>

#if defined(__clang__)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wextra-semi"
#elif defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wextra-semi"
#endif

#include <mdspan/mdarray.hpp>
#include <mdspan/mdspan.hpp>

#if defined(__clang__)
    #pragma clang diagnostic pop
#elif defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

namespace KokkosEx = Kokkos::Experimental;
