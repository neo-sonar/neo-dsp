#pragma once

#include <neo/config.hpp>
#include <neo/container/mdspan.hpp>

#include <utility>

namespace neo {

template<in_vector InVec1, in_vector InVec2, out_vector OutVec>
auto multiply(InVec1 x, InVec2 y, OutVec out) -> void
{
    NEO_EXPECTS(x.extents() == y.extents());
    NEO_EXPECTS(x.extents() == out.extents());

    for (auto i{0}; std::cmp_less(i, x.extent(0)); ++i) {
        out[i] = x[i] * y[i];
    }
}

}  // namespace neo
