#pragma once

#include <neo/config.hpp>

#include <neo/container/mdspan.hpp>

#include <cassert>
#include <concepts>
#include <utility>

namespace neo {

template<in_object InObj1, in_object InObj2, typename BinaryPredicate>
[[nodiscard]] auto allmatch(InObj1 lhs, InObj2 rhs, BinaryPredicate predicate) noexcept -> bool
{
    assert(detail::all_extents_match(lhs, rhs));

    if constexpr (InObj1::rank() == 1) {
        for (auto i{0}; std::cmp_less(i, lhs.extent(0)); ++i) {
            if (not predicate(lhs[i], rhs[i])) {
                return false;
            }
        }
    } else {
        for (auto i{0}; std::cmp_less(i, lhs.extent(0)); ++i) {
            for (auto j{0}; std::cmp_less(j, lhs.extent(1)); ++j) {
                if (not predicate(lhs(i, j), rhs(i, j))) {
                    return false;
                }
            }
        }
    }

    return true;
}

template<in_object InObj, typename UnaryPredicate>
[[nodiscard]] auto allmatch(InObj obj, UnaryPredicate predicate) noexcept -> bool
{
    return allmatch(obj, obj, [predicate](auto val, auto) { return predicate(val); });
}

}  // namespace neo
