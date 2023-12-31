// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

namespace neo {

template<typename T>
using value_type_t = typename T::value_type;

}  // namespace neo
