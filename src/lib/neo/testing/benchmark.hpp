#pragma once

#include <neo/config.hpp>

namespace neo {

template<typename T>
NEO_ALWAYS_INLINE auto do_not_optimize(T& value) -> void
{
#if defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#elif defined(__GNUC__)
    asm volatile("" : "+m,r"(value) : : "memory");
#else
    (void)(value);
#endif
}

}  // namespace neo
