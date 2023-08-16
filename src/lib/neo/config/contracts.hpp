#pragma once

#include <neo/config/preprocessor.hpp>

#include <cassert>
#include <exception>
#include <stdexcept>

namespace neo {

enum struct contracts_check_mode
{
    ignore,
    assertion,
    exception,
};

#if defined(_DEBUG) || !defined(NDEBUG)
inline constexpr auto current_contracts_check_mode = contracts_check_mode::assertion;
#else
inline constexpr auto current_contracts_check_mode = contracts_check_mode::exception;
#endif

}  // namespace neo

#define NEO_EXPECTS(x)                                                                                                 \
    do {                                                                                                               \
        if constexpr (::neo::current_contracts_check_mode == ::neo::contracts_check_mode::exception) {                 \
            if (!(x)) {                                                                                                \
                throw std::runtime_error{"contract violation: " #x " in: '" __FILE__ "'"};                             \
            }                                                                                                          \
        } else if constexpr (::neo::current_contracts_check_mode == ::neo::contracts_check_mode::assertion) {          \
            assert((x));                                                                                               \
        } else {                                                                                                       \
            (void)(x);                                                                                                 \
        }                                                                                                              \
    } while (false)
