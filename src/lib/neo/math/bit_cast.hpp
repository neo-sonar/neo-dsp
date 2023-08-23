#pragma once

#include <bit>

namespace neo {

template<typename To, typename From>
[[nodiscard]] constexpr auto bit_cast(From const& src) noexcept -> To
{
#if defined(__cpp_lib_bit_cast)
    return std::bit_cast<To>(src);
#elif __has_builtin(__builtin_bit_cast)
    return __builtin_bit_cast(To, src);
#else
    #error "constexpr bit_cast is required"
#endif
}

}  // namespace neo
