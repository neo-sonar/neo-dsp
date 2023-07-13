#pragma once

#include <juce_core/juce_core.h>

#include <span>

JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE("-Wextra-semi")
#include <mdspan/mdarray.hpp>
#include <mdspan/mdspan.hpp>
JUCE_END_IGNORE_WARNINGS_GCC_LIKE

namespace KokkosEx = Kokkos::Experimental;
