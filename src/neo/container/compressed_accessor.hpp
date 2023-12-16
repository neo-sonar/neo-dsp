#pragma once

#include <neo/complex/complex.hpp>
#include <neo/container/mdspan.hpp>

#include <concepts>
#include <cstddef>
#include <limits>

namespace neo {

template<typename ElementType, typename NestedAccessor>
struct compressed_accessor
{
    using element_type     = ElementType;
    using reference        = ElementType;
    using data_handle_type = typename NestedAccessor::data_handle_type;
    using offset_policy    = compressed_accessor<ElementType, typename NestedAccessor::offset_policy>;

    constexpr explicit compressed_accessor(NestedAccessor const& a) : _nested_accessor{a} {}

    [[nodiscard]] constexpr auto access(data_handle_type p, size_t i) const noexcept -> reference
    {
        using nested_element_t = typename NestedAccessor::element_type;

        if constexpr (std::floating_point<ElementType>) {
            constexpr auto max_val   = std::numeric_limits<nested_element_t>::max();
            constexpr auto inv_scale = ElementType(1) / static_cast<ElementType>(max_val);
            return static_cast<ElementType>(_nested_accessor.access(p, i)) * inv_scale;
        } else {
            // static_assert(complex<ElementType>);
            using element_real_t     = typename ElementType::value_type;
            using nested_real_t      = typename nested_element_t::value_type;
            constexpr auto max_val   = std::numeric_limits<nested_real_t>::max();
            constexpr auto inv_scale = element_real_t(1) / static_cast<element_real_t>(max_val);

            auto const val = _nested_accessor.access(p, i);
            return ElementType(
                static_cast<element_real_t>(val.real()) * inv_scale,
                static_cast<element_real_t>(val.imag()) * inv_scale
            );
        }
    }

    [[nodiscard]] constexpr auto offset(data_handle_type p, size_t i) const noexcept -> data_handle_type
    {
        return _nested_accessor.offset(p, i);
    }

private:
    NestedAccessor _nested_accessor;
};

}  // namespace neo
