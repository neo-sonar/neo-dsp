#pragma once

#include <neo/container/mdspan.hpp>

namespace neo {

template<in_vector Vec>
struct split_complex
{
    Vec real;
    Vec imag;
};

template<in_vector Vec>
split_complex(Vec re, Vec im) -> split_complex<Vec>;

}  // namespace neo
