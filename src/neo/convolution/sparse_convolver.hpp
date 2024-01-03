// SPDX-License-Identifier: MIT

#pragma once

#include <neo/complex.hpp>
#include <neo/convolution/dense_fdl.hpp>
#include <neo/convolution/overlap_add.hpp>
#include <neo/convolution/overlap_save.hpp>
#include <neo/convolution/sparse_filter.hpp>
#include <neo/convolution/uniform_partitioned_convolver.hpp>

namespace neo {

template<complex Complex>
using sparse_upols_convolver
    = uniform_partitioned_convolver<overlap_save<Complex>, dense_fdl<Complex>, sparse_filter<Complex>>;

template<complex Complex>
using sparse_upola_convolver
    = uniform_partitioned_convolver<overlap_add<Complex>, dense_fdl<Complex>, sparse_filter<Complex>>;

}  // namespace neo
