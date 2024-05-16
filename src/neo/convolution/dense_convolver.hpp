// SPDX-License-Identifier: MIT

#pragma once

#include <neo/config.hpp>

#include <neo/complex.hpp>
#include <neo/convolution/dense_fdl.hpp>
#include <neo/convolution/dense_filter.hpp>
#include <neo/convolution/overlap_add.hpp>
#include <neo/convolution/overlap_add_convolver.hpp>
#include <neo/convolution/overlap_save.hpp>
#include <neo/convolution/uniform_partitioned_convolver.hpp>
#include <neo/type_traits/value_type_t.hpp>

namespace neo::convolution {

/// \ingroup neo-convolution
template<complex Complex>
using upols_convolver = uniform_partitioned_convolver<overlap_save<Complex>, dense_fdl<Complex>, dense_filter<Complex>>;

/// \ingroup neo-convolution
template<complex Complex>
using upola_convolver = uniform_partitioned_convolver<overlap_add<Complex>, dense_fdl<Complex>, dense_filter<Complex>>;

/// \ingroup neo-convolution
template<neo::complex Complex>
using upola_convolver_v2 = overlap_add_convolver<Complex, dense_fdl<Complex>, dense_filter<Complex>>;

/// \ingroup neo-convolution
template<complex Complex>
using split_upola_convolver = uniform_partitioned_convolver<
    overlap_add<Complex>,
    dense_split_fdl<value_type_t<Complex>>,
    dense_split_filter<value_type_t<Complex>>>;

/// \ingroup neo-convolution
template<complex Complex>
using split_upols_convolver = uniform_partitioned_convolver<
    overlap_save<Complex>,
    dense_split_fdl<value_type_t<Complex>>,
    dense_split_filter<value_type_t<Complex>>>;

}  // namespace neo::convolution
