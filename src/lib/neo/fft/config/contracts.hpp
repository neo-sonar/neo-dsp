#pragma once

#include <neo/fft/config/preprocessor.hpp>

#include <cassert>
#include <exception>
#include <stdexcept>

namespace neo::fft {

enum struct contracts_check_mode
{
    ignore,
    assertion,
    exception,
};

#if defined(NDEBUG)
inline constexpr auto current_contracts_check_mode = contracts_check_mode::assertion;
#else
inline constexpr auto current_contracts_check_mode = contracts_check_mode::exception;
#endif

}  // namespace neo::fft

#define NEO_FFT_PRECONDITION(x)                                                                                         \
    do {                                                                                                                \
        if constexpr (::neo::fft::current_contracts_check_mode == ::neo::fft::contracts_check_mode::exception) {        \
            if (!(x)) {                                                                                                 \
                throw std::runtime_error{"contract violation: " #x};                                                    \
            }                                                                                                           \
        } else if constexpr (::neo::fft::current_contracts_check_mode == ::neo::fft::contracts_check_mode::assertion) { \
            assert((x));                                                                                                \
        } else {                                                                                                        \
            (void)(x);                                                                                                  \
        }                                                                                                               \
    } while (false)
