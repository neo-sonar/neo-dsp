#pragma once

#include <neo/container/mdspan.hpp>

#include <concepts>
#include <cstddef>
#include <limits>

namespace neo {

template<typename ElementType, typename NestedAccessor>
    requires std::signed_integral<typename NestedAccessor::element_type>
struct compressed_float_accessor
{
    using element_type     = ElementType;
    using reference        = ElementType;
    using data_handle_type = typename NestedAccessor::data_handle_type;
    using offset_policy    = compressed_float_accessor<ElementType, typename NestedAccessor::offset_policy>;

    constexpr compressed_float_accessor(NestedAccessor const& a) : _nested_accessor{a} {}

    [[nodiscard]] constexpr auto access(data_handle_type p, size_t i) const noexcept -> reference
    {
        return static_cast<ElementType>(_nested_accessor.access(p, i)) * inv_scale;
    }

    [[nodiscard]] constexpr auto offset(data_handle_type p, size_t i) const noexcept -> data_handle_type
    {
        return _nested_accessor.offset(p, i);
    }

private:
    using nested_element_t          = typename NestedAccessor::element_type;
    static constexpr auto max_val   = std::numeric_limits<nested_element_t>::max();
    static constexpr auto scale     = static_cast<ElementType>(max_val);
    static constexpr auto inv_scale = ElementType(1) / scale;

    NestedAccessor _nested_accessor;
};

}  // namespace neo
